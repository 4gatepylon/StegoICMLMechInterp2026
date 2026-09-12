# ruff: noqa: F722, F821  # jaxtyping shape strings are not Python expressions.
"""The small TRL trainer used by both LoRA distillation stages."""

import os
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from datasets import Dataset
from jaxtyping import Float
from torch import Tensor
from trl import SFTConfig, SFTTrainer

from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.artifacts import save_experiment_config
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.colors import (
    ColorPartition,
    build_color_partition,
    tokenize_prefixes,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.constants import GREEN_SIGNAL, RED_SIGNAL
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.data import CorpusConfig, CorpusSplits, TextExample
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.models import (
    clear_device_cache,
    load_trainable_lora_model,
)


class ShiftDistillationTrainer(SFTTrainer):
    """Distill fixed vocabulary shifts into prefix-conditioned LoRA weights."""

    def __init__(
        self,
        *args: Any,
        signals: Sequence[int],
        prefix_ids: dict[int, tuple[int, ...]],
        partition: ColorPartition,
        delta: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not signals:
            raise ValueError("At least one signal is required")
        if delta <= 0:
            raise ValueError("delta must be positive")

        self.signals = tuple(signals)
        self.prefix_ids = {signal: torch.tensor(prefix_ids[signal], dtype=torch.long) for signal in self.signals}
        self.delta_vectors: dict[int, Float[Tensor, "vocab"]] = {}
        for signal in self.signals:
            shift: Float[Tensor, "vocab"] = torch.zeros(self.model.config.vocab_size, dtype=torch.float32)
            if signal == RED_SIGNAL:
                shift[list(partition.red_ids)] = delta
            elif signal == GREEN_SIGNAL:
                shift[list(partition.green_ids)] = delta
            self.delta_vectors[signal] = shift

        # This loss performs its own token normalization.
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | None = None,
    ) -> Float[Tensor, ""] | tuple[Float[Tensor, ""], Any]:
        raw = {key: inputs[key] for key in ("input_ids", "attention_mask") if key in inputs}
        raw["use_cache"] = False

        peft_model = self.accelerator.unwrap_model(model)
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad(), peft_model.disable_adapter():
                teacher_logits: Float[Tensor, "batch token vocab"] = model(**raw).logits[:, :-1, :].float()
        finally:
            model.train(was_training)

        valid = inputs["labels"][:, 1:].ne(-100)
        losses = []
        outputs = None
        for signal in self.signals:
            prefix = self.prefix_ids[signal].to(inputs["input_ids"].device)
            prefix = prefix.unsqueeze(0).expand(inputs["input_ids"].shape[0], -1)
            student_inputs = {
                "input_ids": torch.cat((prefix, inputs["input_ids"]), dim=1),
                "use_cache": False,
            }
            if "attention_mask" in inputs:
                prefix_mask = torch.ones_like(prefix)
                student_inputs["attention_mask"] = torch.cat((prefix_mask, inputs["attention_mask"]), dim=1)

            outputs = model(**student_inputs)
            prefix_length = prefix.shape[1]
            student_logits: Float[Tensor, "batch token vocab"] = outputs.logits[:, prefix_length : prefix_length + teacher_logits.shape[1], :].float()
            log_q: Float[Tensor, "batch token vocab"] = F.log_softmax(
                teacher_logits + self.delta_vectors[signal].to(teacher_logits.device),
                dim=-1,
            )
            log_p: Float[Tensor, "batch token vocab"] = F.log_softmax(student_logits, dim=-1)
            token_kl: Float[Tensor, "batch token"] = F.kl_div(log_p, log_q, log_target=True, reduction="none").sum(dim=-1)
            losses.append((token_kl * valid).sum() / valid.sum().clamp_min(1))

        loss: Float[Tensor, ""] = torch.stack(losses).mean()
        return (loss, outputs) if return_outputs else loss


def _dataset(examples: Sequence[TextExample]) -> Dataset:
    return Dataset.from_dict({"input_ids": [list(example.input_ids) for example in examples]})


def _configure_tracking(args: Any, stage: str, output_dir: Path) -> None:
    reports = list(args.trainer_config.report_to)
    if args.wandb_mode == "disabled":
        args.trainer_config.report_to = [name for name in reports if name != "wandb"]
        return
    if "wandb" not in reports:
        args.trainer_config.report_to = [*reports, "wandb"]
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ["WANDB_DIR"] = str(output_dir)
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    args.trainer_config.run_name = f"{args.wandb_run_name}-{stage}"
    os.environ["WANDB_RUN_GROUP"] = args.wandb_run_name


def _trainer_dtype(config: SFTConfig) -> torch.dtype:
    if config.bf16:
        return torch.bfloat16
    if config.fp16:
        return torch.float16
    return torch.float32


def run_training_stage(
    *,
    args: Any,
    tokenizer: Any,
    corpus_config: CorpusConfig,
    splits: CorpusSplits,
    train_examples: Sequence[TextExample],
    signals: Sequence[int],
    stage: str,
    adapter_path: str | None,
) -> None:
    """Load and train one stage; optimization and logging come from SFTConfig."""

    device = args.trainer_config.device
    dtype = _trainer_dtype(args.trainer_config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    args.trainer_config.output_dir = str(output_dir)
    args.trainer_config.logging_dir = str(output_dir / "runs")
    _configure_tracking(
        args,
        f"stage{1 if stage == 'prefix' else 2}-{stage}",
        output_dir,
    )

    source = f" {adapter_path!r}" if adapter_path else ""
    print(f"Loading trainable LoRA model{source}...", flush=True)
    model, base_model_name = load_trainable_lora_model(
        args.model_spec,
        dtype,
        adapter_path=adapter_path,
        lora_config=args.lora_config,
    )
    partition = build_color_partition(tokenizer, model.config.vocab_size, seed=args.vocab_seed)
    save_experiment_config(
        output_dir,
        stage=stage,
        args=args,
        corpus_config=corpus_config,
        splits=splits,
        base_model_name=base_model_name,
    )

    validation_examples = splits.validation[: args.max_eval_sequences]
    trainer = ShiftDistillationTrainer(
        model=model,
        args=args.trainer_config,
        train_dataset=_dataset(train_examples),
        eval_dataset=_dataset(validation_examples),
        processing_class=tokenizer,
        signals=signals,
        prefix_ids=tokenize_prefixes(tokenizer),
        partition=partition,
        delta=args.delta,
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model()
    trainer.save_state()
    trainer.accelerator.unwrap_model(model).to("cpu")
    clear_device_cache(device)
    print(f"Saved {stage} adapter to {output_dir}", flush=True)
