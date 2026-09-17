"""Minimal gated red/green KL trainer."""

import json
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterator, Literal, override

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl import SFTTrainer

from ciphers.kirchenbauer_et_al.src.data_kl_fineweb import TokenBatch, prefix_batch, tokenize_with_prefix

N_BITS = 8
DELTA = 1.0
STRATEGY = "block"

StudentLogprobs = Float[torch.Tensor, "batch prefixed_tokens vocab"]  # noqa: F722
TargetLogprobs = Float[torch.Tensor, "batch free_tokens vocab"]  # noqa: F722
PrefixTargets = Int[torch.Tensor, "batch prefix_tokens"]  # noqa: F722
ScalarLoss = Float[torch.Tensor, ""]  # noqa: F722
TokenPositions = Int[torch.Tensor, "positions"]  # noqa: F821


class InputDumpCallback(TrainerCallback):
    """Write the first ``limit`` training microbatches per rank (zero disables).

    ``PrefixKLTrainer`` calls ``dump`` with the exact teacher/student forward
    dictionaries; standard Trainer callbacks do not receive model inputs.
    Each train invocation replaces this rank's JSONL under the artifact-backed
    Trainer output directory, including when resuming from a checkpoint.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.count = self.accumulation_step = 0

    @override
    def on_train_begin(self, args, state, control, **kwargs) -> None:
        """Replace the parent's no-op with file initialization and a stdout path."""
        self.count = self.accumulation_step = 0
        if self.limit:
            directory = Path(args.output_dir) / "input_dumps"
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / f"rank-{args.process_index}.jsonl"
            self.path.write_text("", encoding="utf-8")
            print(f"[input dump] rank {args.process_index}: first {self.limit} training microbatches -> {self.path}", flush=True)

    @override
    def on_step_begin(self, args, state, control, **kwargs) -> None:
        """Reset accumulation position at each optimizer step, unlike the no-op parent."""
        self.accumulation_step = 0

    def dump(self, trainer, teacher: dict[str, TokenBatch], student: dict[str, TokenBatch], prefix_length: int) -> None:
        """Append one inspection record; return None and leave tensors unchanged.

        ``trainer`` supplies tokenizer, rank, state and actual accumulation size.
        ``teacher`` and ``student`` each require ``input_ids`` and
        ``attention_mask`` tensors [batch, tokens], exactly as passed to forward.
        ``prefix_length`` counts the student's leading control tokens.
        JSONL records contain rank/world_size, zero-based completed global_step,
        one-based optimizer_step, microbatch and gradient_accumulation_step,
        gradient_accumulation_steps, prefix_length, and tokenizer metadata.
        Each teacher/student object holds both input arrays, shape, decoded
        strings (special tokens retained, whitespace cleanup disabled), and
        per-row padding counts derived from mask zeros. A PAD ID with mask 1
        is a real token (e.g. shared EOS/PAD), not padding.
        Evaluation does not consume the limit. Requires on_train_begin and
        on_step_begin events from Trainer before training forwards.
        """
        if not self.limit or self.count >= self.limit or not trainer.model.training:
            return
        self.count += 1
        self.accumulation_step += 1
        tokenizer = trainer.processing_class
        record = {
            "rank": trainer.args.process_index,
            "world_size": trainer.args.world_size,
            "global_step": trainer.state.global_step,
            "optimizer_step": trainer.state.global_step + 1,
            "microbatch": self.count,
            "gradient_accumulation_step": self.accumulation_step,
            "gradient_accumulation_steps": trainer.current_gradient_accumulation_steps,
            "prefix_length": prefix_length,
            "tokenizer": {
                key: getattr(tokenizer, key) for key in ("name_or_path", "padding_side", "pad_token", "pad_token_id", "bos_token", "bos_token_id", "eos_token", "eos_token_id")
            },
        }
        for name, model_inputs in (("teacher", teacher), ("student", student)):
            arrays = {key: value.detach().cpu().tolist() for key, value in model_inputs.items()}
            record[name] = {
                **arrays,
                "shape": list(model_inputs["input_ids"].shape),
                "decoded": tokenizer.batch_decode(arrays["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                "padding_tokens_per_row": [mask.count(0) for mask in arrays["attention_mask"]],
            }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class LossInformation:
    """Detached scalar loss components retained only for metric logging.

    Attributes:
        prefix_loss: Mean prefix-token NLL after any mode-specific behavior.
            Shape ``[]``. ``_divergence()`` detaches this tensor from autograd.
        data_loss: Mean free-token KL after applying mode-specific weighting.
            Shape ``[]``. ``_divergence()`` detaches this tensor from autograd.

    ``PrefixKLTrainer._record_loss_metrics()`` gathers both fields across
    processes and converts them to Python floats. Callers must optimize the
    separate total loss returned by ``_divergence()``.
    """

    prefix_loss: ScalarLoss
    data_loss: ScalarLoss


def padded_input_tokens_seen(
    global_step: int,
    max_length: int,
    local_batch_size: int,
    gradient_accumulation_steps: int,
    process_count: int,
) -> int:
    """Return cumulative fixed-shape student input tokens at an optimizer step.

    Args:
        global_step: Number of completed optimizer updates, including updates
            restored from a checkpoint.
        max_length: Padded sequence width produced by the training collator.
        local_batch_size: Number of examples processed by each process in one
            microbatch, including all devices managed within that process.
        gradient_accumulation_steps: Microbatches consumed per optimizer update.
        process_count: Distributed processes contributing distinct microbatches.

    Returns:
        The number of padded student-input token slots consumed through
        ``global_step``. ``PrefixKLTrainer.log()`` reports this cumulative value
        as ``num_padded_input_tokens_seen``. The calculation assumes the
        trainer's fixed-size, batch-aligned iterable dataset contract; unlike
        ``num_input_tokens_seen``, it deliberately includes padding and does not
        count the separate adapter-disabled teacher forward pass.
    """
    return global_step * max_length * local_batch_size * gradient_accumulation_steps * process_count


def prefix_bits_encoding_text_collator(
    examples: list[dict[str, object]],
    tokenizer,
    n_bits: int,
    data_length: int,
) -> dict[str, object]:
    """Build fixed-width data plus prefix batches consumed by PrefixKLTrainer.

    ``examples`` contains ``text`` strings and optionally both ``prefix_bits``
    (an ``n_bits``-wide binary string) and ``do_encoding`` (a boolean) on every
    row. Otherwise these controls are sampled. ``tokenizer`` is passed to
    ``tokenize_with_prefix``, which concatenates prefix and data token IDs.
    ``data_length`` is the padded document width, excluding the prefix, and
    must be positive and divisible by ``n_bits``. The returned dictionary's
    complete schema is documented at its consumer, ``PrefixKLTrainer.compute_loss``.
    """
    if n_bits < 1 or data_length < 1 or data_length % n_bits:
        raise ValueError("data_length must be positive and divisible by positive n_bits")
    texts = [example["text"] for example in examples]
    has_fixed_prefix = ["prefix_bits" in example or "do_encoding" in example for example in examples]
    if any(has_fixed_prefix):
        if not all("prefix_bits" in example and "do_encoding" in example for example in examples):
            raise ValueError("fixed prefix metadata must be present on every example in a batch")
        bits = [example["prefix_bits"] for example in examples]
        enabled = [example["do_encoding"] for example in examples]
    else:
        _, bits, enabled = prefix_batch(texts, n_bits)
    if any(len(bit) != n_bits for bit in bits):
        raise ValueError("prefix_bits must contain exactly n_bits bits")
    prefixed_model_inputs, unprefixed_model_inputs, Q = tokenize_with_prefix(tokenizer, texts, bits, enabled, data_length)
    auxiliary_inputs = {
        # A non-None `labels` key makes prediction_step call our compute_loss; -100 marks padding.
        # Our loss ignores `labels`; the default causal-LM loss uses shifted targets and skips -100.
        "labels": prefixed_model_inputs["input_ids"].masked_fill(prefixed_model_inputs["attention_mask"] == 0, -100),
        "base_input_ids": unprefixed_model_inputs["input_ids"],
        "base_attention_mask": unprefixed_model_inputs["attention_mask"],
        "prefix_bits": bits,
        "do_encoding": enabled,
        "prefix_length": Q,
    }
    assert prefixed_model_inputs.keys().isdisjoint(auxiliary_inputs)
    return {**prefixed_model_inputs, **auxiliary_inputs}


def prefix_nll(
    student_logprobs: StudentLogprobs,
    prefix_targets: PrefixTargets,
    Q: int,
) -> ScalarLoss:
    """Compute NLL on the prefix tokens that communicate the hidden message.

    Args:
        student_logprobs: Log-probabilities for each token in the prefixed model
            input, with shape ``[batch, Q + free_tokens, vocab]``.
        prefix_targets: Next-token targets for the prefix, with shape
            ``[batch, Q]``.
        Q: Number of prefix positions. This must equal
            ``prefix_targets.shape[1]``.

    Returns:
        A scalar mean negative log-likelihood. ``PrefixKLTrainer`` uses this
        value as both a loss component and a separately logged metric.
    """
    return -student_logprobs[:, :Q].gather(-1, prefix_targets[:, :, None]).squeeze(-1).mean()


def free_token_kl(
    student_logprobs: StudentLogprobs,
    target_logprobs: TargetLogprobs,
    Q: int,
) -> ScalarLoss:
    """Compute KL divergence on non-prefix tokens carrying the encoded data.

    Args:
        student_logprobs: Log-probabilities for each token in the prefixed model
            input, with shape ``[batch, Q + free_tokens, vocab]``.
        target_logprobs: Teacher log-probabilities after applying the encoding
            gates, with shape ``[batch, free_tokens, vocab]``.
        Q: Number of leading student positions to exclude so the remaining
            positions align with ``target_logprobs``.

    Returns:
        A scalar mean KL divergence. ``PrefixKLTrainer`` applies any mode-specific
        weighting and logs the resulting data-loss component.
    """
    return F.kl_div(student_logprobs[:, Q:], target_logprobs.exp(), reduction="none").sum(-1).mean()


class PrefixKLTrainer(SFTTrainer):
    """Train LoRA logits toward gated boosts, optionally learning the prefix with NLL.

    ``reject_document_padding=True`` rejects any masked token before teacher or
    student execution. Set it false only to explicitly permit right padding;
    left/internal padding is always forbidden because it shifts bit positions.
    The required collator supplies both masks; the opt-out retains the historical
    loss over all data slots, including pads, rather than changing the objective.
    """

    def __init__(
        self,
        *args,
        loss_mode: Literal["nll", "ignore_prefix"] = "nll",
        alpha: float = 1.0,
        n_bits: int = N_BITS,
        delta: float = DELTA,
        strategy: Literal["block", "modulo"] = STRATEGY,
        profile_memory_steps: int = 0,
        reject_document_padding: bool = True,
        dump_inputs: int = 1,
        data_collator=None,
        **kwargs,
    ) -> None:
        if loss_mode not in {"nll", "ignore_prefix"}:
            raise ValueError("loss_mode must be 'nll' or 'ignore_prefix'")
        if n_bits < 1 or strategy not in {"block", "modulo"}:
            raise ValueError("n_bits must be positive and strategy must be 'block' or 'modulo'")
        if profile_memory_steps < 0:
            raise ValueError("profile_memory_steps must be nonnegative")
        collator_function = data_collator.func if isinstance(data_collator, partial) else data_collator
        if collator_function is not prefix_bits_encoding_text_collator:
            raise ValueError("PrefixKLTrainer requires prefix_bits_encoding_text_collator")
        self.loss_mode, self.alpha, self.n_bits, self.delta, self.strategy = loss_mode, alpha, n_bits, delta, strategy
        self.reject_document_padding = reject_document_padding
        self.profile_memory_steps, self._profile_calls, self._profile_this_call = profile_memory_steps, 0, False
        super().__init__(*args, data_collator=data_collator, **kwargs)
        self.input_dump_callback = InputDumpCallback(dump_inputs)
        self.add_callback(self.input_dump_callback)
        # This loss ignores num_items_in_batch, so retain "default batch size reduction":
        # https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/trainer#transformers.Trainer.compute_loss
        self.model_accepts_loss_kwargs = False

    def _validate_padding(self, attention_mask: TokenBatch, *, name: str) -> None:
        """Reject masks that would assign watermark bits to padding positions.

        Args:
            attention_mask: Binary mask of shape ``[batch, tokens]`` from the
                required collator. One denotes real tokens, zero padding.
            name: Input key used in errors to distinguish teacher and student
                masks. Both are checked independently before either forward.

        Returns:
            None if the mask satisfies the policy. Left padding and internal
            holes always raise ValueError. Right padding also raises when
            ``reject_document_padding`` is true (the default). Disabling that
            guard restores the previous loss behavior, which includes padded
            positions in the bit partition and KL; it does not mask them away.
            Token IDs are not inspected because a pad ID can also be a real EOS.
        """
        if torch.any(attention_mask[:, 0] == 0) or torch.any(attention_mask[:, 1:] > attention_mask[:, :-1]):
            raise ValueError(f"{name} contains left or internal padding, which is always forbidden")
        if self.reject_document_padding and torch.any(attention_mask == 0):
            raise ValueError(
                f"{name} contains right padding; every document must fill data_length. Filter by Qwen length >= data_length or explicitly set reject_document_padding=False."
            )

    def _positions(self, part: int, n_tokens: int, device: torch.device) -> TokenPositions:
        part_size = n_tokens // self.n_bits
        return torch.arange(part * part_size, (part + 1) * part_size, device=device) if self.strategy == "block" else torch.arange(part, n_tokens, self.n_bits, device=device)

    @staticmethod
    def _color(bit: str, vocab_size: int) -> slice:
        midpoint = vocab_size // 2
        return slice(0, midpoint) if bit == "0" else slice(midpoint, None)

    def _divergence(
        self,
        student_logprobs: StudentLogprobs,
        target_logprobs: TargetLogprobs,
        prefix_targets: PrefixTargets,
        Q: int,
    ) -> tuple[ScalarLoss, LossInformation]:
        """Compose the training objective while retaining its logged components.

        Args:
            student_logprobs: Log-probabilities for the prefixed input, with
                shape ``[batch, Q + free_tokens, vocab]``.
            target_logprobs: Encoded teacher log-probabilities, with shape
                ``[batch, free_tokens, vocab]``.
            prefix_targets: Next-token targets for the prefix, with shape
                ``[batch, Q]``.
            Q: Number of prefix positions separating prefix and free tokens.

        Returns:
            The attached scalar total loss and a ``LossInformation`` containing
            detached scalar prefix and data losses. ``compute_loss()`` optimizes
            the total and passes the bundle to ``_record_loss_metrics()``.
        """
        unweighted_data_loss = free_token_kl(student_logprobs, target_logprobs, Q)
        if self.loss_mode == "nll":
            prefix_loss = prefix_nll(student_logprobs, prefix_targets, Q)
            data_loss = self.alpha * unweighted_data_loss
        else:
            prefix_loss = unweighted_data_loss.new_zeros(())
            data_loss = unweighted_data_loss
        total_loss = prefix_loss + data_loss
        loss_information = LossInformation(
            prefix_loss=prefix_loss.detach(),
            data_loss=data_loss.detach(),
        )
        return total_loss, loss_information

    def _record_loss_metrics(self, loss_information: LossInformation) -> None:
        """Gather detached components for SFTTrainer's next log event."""
        mode = "train" if self.model.training else "eval"
        for name, value in (
            ("prefix_loss", loss_information.prefix_loss),
            ("data_loss", loss_information.data_loss),
        ):
            gathered_value = self.accelerator.gather_for_metrics(value).mean().item()
            self._metrics[mode][name].append(gathered_value)

    @override
    def _save_checkpoint(self, model: torch.nn.Module, trial: object | None) -> None:
        """Run the parent save, then visibly report completion on the saving process.

        model and trial are forwarded unchanged to the parent's checkpoint logic.
        Unlike its pre-save INFO message, this flushed console line appears only
        after success, regardless of logging verbosity. The path uses the parent's
        trial-aware output directory. Returns None; save failures propagate without
        a success message, and non-saving ranks remain silent.
        """
        super()._save_checkpoint(model, trial)
        if self.args.should_save:
            checkpoint = Path(self._get_output_dir(trial=trial)) / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            print(f"Saved checkpoint at step {self.state.global_step}: {checkpoint}", flush=True)

    @override
    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Add a cumulative padded-token count before normal SFT logging.

        The parent implementation logs the native ``num_input_tokens_seen``
        counter according to ``include_num_input_tokens_seen`` and merges the
        buffered loss metrics. This override additionally reports fixed-width
        student input slots as ``num_padded_input_tokens_seen``. It derives that
        count from completed optimizer steps, so checkpoint-restored
        ``global_step`` keeps it cumulative across resumed training.
        """
        logs["num_padded_input_tokens_seen"] = padded_input_tokens_seen(
            global_step=self.state.global_step,
            max_length=self.args.max_length,
            local_batch_size=self.args.train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            process_count=self.accelerator.num_processes,
        )
        super().log(logs, start_time)

    def _profile_memory(self, stage: str) -> None:
        if not self._profile_this_call or self.accelerator.device.type != "cuda":
            return
        device = self.accelerator.device
        free, total = torch.cuda.mem_get_info(device)
        gib = 2**30
        print(
            f"[rank {self.accelerator.process_index}] {stage}: "
            f"allocated={torch.cuda.memory_allocated(device) / gib:.2f} GiB, "
            f"reserved={torch.cuda.memory_reserved(device) / gib:.2f} GiB, "
            f"peak={torch.cuda.max_memory_allocated(device) / gib:.2f} GiB, "
            f"free={free / gib:.2f}/{total / gib:.2f} GiB",
            flush=True,
        )

    @contextmanager
    def _memory_stage(self, stage: str) -> Iterator[None]:
        try:
            yield
        except torch.OutOfMemoryError:
            self._profile_memory(f"{stage} OOM")
            raise
        self._profile_memory(f"{stage} ready")

    @override
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ) -> ScalarLoss | tuple[ScalarLoss, object]:
        """Replace SFTTrainer's causal-LM loss with the gated prefix KL objective.

        Unlike the parent loss, this checks both attention masks before either
        model forward: left/internal padding is forbidden, and right padding
        fails unless ``reject_document_padding=False`` was explicitly selected.

        ``prefix_bits_encoding_text_collator()`` produces these required fields::

            {
                "input_ids": Tensor[B, Q + M],         # Prefixed model token IDs.
                "attention_mask": Tensor[B, Q + M],    # Mask for the prefixed model input.
                "labels": Tensor[B, Q + M],            # Trainer routing labels; unused by this loss.
                "base_input_ids": Tensor[B, M],        # Unprefixed teacher token IDs.
                "base_attention_mask": Tensor[B, M],   # Mask for the teacher input.
                "prefix_bits": list[str],              # B messages that select token-color boosts.
                "do_encoding": list[bool],             # Whether to apply boosts to each example.
                "prefix_length": int,                  # Q, used to align prefixed and teacher logits.
            }
        """
        self._validate_padding(inputs["base_attention_mask"], name="base_attention_mask")
        self._validate_padding(inputs["attention_mask"], name="attention_mask")
        self._profile_this_call = self._profile_calls < self.profile_memory_steps
        self._profile_calls += 1
        if self._profile_this_call and self.accelerator.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.accelerator.device)
        self._profile_memory("start")
        # These fields are documented in this function's docstring.
        bits, enabled, Q = inputs["prefix_bits"], inputs["do_encoding"], inputs["prefix_length"]
        M = inputs["base_input_ids"].shape[1]
        device = self.accelerator.device
        unprefixed_model_inputs = {"input_ids": inputs["base_input_ids"], "attention_mask": inputs["base_attention_mask"]}
        prefixed_model_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        self.input_dump_callback.dump(self, unprefixed_model_inputs, prefixed_model_inputs, Q)
        self._profile_memory("inputs ready")

        # TODO(hadriano): Profile peak memory here: dense [B, T, V] teacher/student logits,
        # Accelerate's BF16-to-FP32 output cast, and unreduced KL intermediates are the likely
        # bottleneck; evaluate chunked logits/loss to understand and fix it.
        with self._memory_stage("teacher logprobs"):
            with torch.no_grad(), self.model.disable_adapter():
                teacher_logprobs = model(**unprefixed_model_inputs).logits.log_softmax(dim=-1)
        with self._memory_stage("student logits"):
            outputs = model(**prefixed_model_inputs)
        with self._memory_stage("student logprobs"):
            student_logprobs = outputs.logits.log_softmax(dim=-1)
        prefix_targets = prefixed_model_inputs["input_ids"][:, 1 : Q + 1]
        target_logprobs = teacher_logprobs

        for row, (gate, message) in enumerate(zip(enabled, bits)):
            if not gate:
                continue
            for j, bit in enumerate(message):
                positions = self._positions(j, M, device)
                color = self._color(bit, target_logprobs.shape[-1])
                target_logprobs[row, positions, color] += self.delta

        with self._memory_stage("target logprobs"):
            target_logprobs = target_logprobs.log_softmax(dim=-1)
        with self._memory_stage("loss"):
            loss, loss_information = self._divergence(student_logprobs, target_logprobs, prefix_targets, Q)
        self._record_loss_metrics(loss_information)
        return (loss, outputs) if return_outputs else loss
