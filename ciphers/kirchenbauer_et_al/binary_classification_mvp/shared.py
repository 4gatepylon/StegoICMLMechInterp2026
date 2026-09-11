"""Shared utilities for the fixed red/green policy experiment."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen3-4B-Base"
DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_CONFIG = "sample-10BT"
DATASET_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_PREFIX_OUTPUT = EXPERIMENT_DIR / "outputs" / "prefix_adapter"
DEFAULT_ENCODING_OUTPUT = EXPERIMENT_DIR / "outputs" / "encoding_adapter"

NULL_SIGNAL = -1
RED_SIGNAL = 0
GREEN_SIGNAL = 1
SIGNAL_NAMES = {
    NULL_SIGNAL: "none",
    RED_SIGNAL: "red",
    GREEN_SIGNAL: "green",
}

PREFIXES = {
    NULL_SIGNAL: ("<encoding> <do_encoding> no </do_encoding> <encoding_value> none </encoding_value> </encoding>\n"),
    RED_SIGNAL: ("<encoding> <do_encoding> yes </do_encoding> <encoding_value> 0 </encoding_value> </encoding>\n"),
    GREEN_SIGNAL: ("<encoding> <do_encoding> yes </do_encoding> <encoding_value> 1 </encoding_value> </encoding>\n"),
}


@dataclass(frozen=True)
class CorpusConfig:
    """Configuration that must agree across all three scripts."""

    prefix_train_tokens: int = 65_536
    encoding_train_tokens: int = 65_536
    validation_tokens: int = 16_384
    generation_prompts: int = 256
    sequence_length: int = 256
    generation_prompt_length: int = 128
    min_sequence_length: int = 32
    dataset_seed: int = 42
    shuffle_buffer_size: int = 10_000


@dataclass(frozen=True)
class TextExample:
    """One tokenized FineWeb document fragment."""

    input_ids: tuple[int, ...]
    source_hash: str

    @property
    def prediction_tokens(self) -> int:
        return max(len(self.input_ids) - 1, 0)


@dataclass(frozen=True)
class CorpusSplits:
    """Four source-document-disjoint FineWeb splits."""

    prefix_train: tuple[TextExample, ...]
    encoding_train: tuple[TextExample, ...]
    validation: tuple[TextExample, ...]
    generation: tuple[TextExample, ...]

    def assert_disjoint(self) -> None:
        split_hashes = {
            "prefix_train": {item.source_hash for item in self.prefix_train},
            "encoding_train": {item.source_hash for item in self.encoding_train},
            "validation": {item.source_hash for item in self.validation},
            "generation": {item.source_hash for item in self.generation},
        }
        names = list(split_hashes)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = split_hashes[left] & split_hashes[right]
                if overlap:
                    raise RuntimeError(f"FineWeb splits {left!r} and {right!r} overlap: {len(overlap)} source documents")


@dataclass(frozen=True)
class ColorPartition:
    """A deterministic partition of non-special vocabulary IDs."""

    green_ids: tuple[int, ...]
    red_ids: tuple[int, ...]
    special_ids: tuple[int, ...]
    seed: int


@dataclass
class DistributionMetrics:
    """Differentiable loss plus detached summary statistics."""

    loss: torch.Tensor
    expected_red: float
    expected_green: float
    expected_uncolored: float
    token_count: int

    def as_record(self, *, signal: int, step: int, split: str) -> dict[str, Any]:
        denominator = max(self.token_count, 1)
        return {
            "step": step,
            "split": split,
            "signal": SIGNAL_NAMES[signal],
            "kl": float(self.loss.detach().cpu()),
            "expected_red": self.expected_red,
            "expected_green": self.expected_green,
            "expected_uncolored": self.expected_uncolored,
            "red_rate": self.expected_red / denominator,
            "green_rate": self.expected_green / denominator,
            "token_count": self.token_count,
        }


def add_corpus_arguments(parser: Any) -> None:
    defaults = CorpusConfig()
    parser.add_argument(
        "--prefix-train-tokens",
        type=int,
        default=defaults.prefix_train_tokens,
    )
    parser.add_argument(
        "--encoding-train-tokens",
        type=int,
        default=defaults.encoding_train_tokens,
    )
    parser.add_argument(
        "--validation-tokens",
        type=int,
        default=defaults.validation_tokens,
    )
    parser.add_argument(
        "--generation-prompts",
        type=int,
        default=defaults.generation_prompts,
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=defaults.sequence_length,
    )
    parser.add_argument(
        "--generation-prompt-length",
        type=int,
        default=defaults.generation_prompt_length,
    )
    parser.add_argument(
        "--min-sequence-length",
        type=int,
        default=defaults.min_sequence_length,
    )
    parser.add_argument("--dataset-seed", type=int, default=defaults.dataset_seed)
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=defaults.shuffle_buffer_size,
    )


def add_training_arguments(parser: Any) -> None:
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--vocab-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logit-chunk-size", type=int, default=32)
    parser.add_argument("--log-every-steps", type=int, default=8)
    parser.add_argument("--eval-every-steps", type=int, default=32)
    parser.add_argument("--max-eval-sequences", type=int, default=16)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional smoke-test cap; by default all configured epochs are run.",
    )


def corpus_config_from_args(args: Any) -> CorpusConfig:
    return CorpusConfig(
        prefix_train_tokens=args.prefix_train_tokens,
        encoding_train_tokens=args.encoding_train_tokens,
        validation_tokens=args.validation_tokens,
        generation_prompts=args.generation_prompts,
        sequence_length=args.sequence_length,
        generation_prompt_length=args.generation_prompt_length,
        min_sequence_length=args.min_sequence_length,
        dataset_seed=args.dataset_seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )


def _source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_corpus_splits(tokenizer: Any, config: CorpusConfig) -> CorpusSplits:
    """Stream FineWeb once and allocate whole documents to disjoint splits.

    Each accepted document contributes at most one truncated fragment. When a
    token budget is met, the remainder of the current document is discarded
    before allocation begins for the next split. Exact-text hashes are also
    de-duplicated across the stream.
    """

    if config.min_sequence_length < 2:
        raise ValueError("min_sequence_length must be at least 2")
    if config.min_sequence_length > config.sequence_length:
        raise ValueError("min_sequence_length cannot exceed sequence_length")
    if config.generation_prompt_length < 1:
        raise ValueError("generation_prompt_length must be positive")

    stream = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIG,
        split="train",
        streaming=True,
        revision=DATASET_REVISION,
    ).shuffle(
        seed=config.dataset_seed,
        buffer_size=config.shuffle_buffer_size,
    )
    iterator = iter(stream)
    seen_hashes: set[str] = set()

    def next_example(max_length: int, min_length: int) -> TextExample:
        while True:
            row = next(iterator)
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            digest = _source_hash(text)
            if digest in seen_hashes:
                continue
            token_ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
            if len(token_ids) < min_length:
                continue
            seen_hashes.add(digest)
            return TextExample(tuple(token_ids), digest)

    def collect_training_split(token_budget: int) -> tuple[TextExample, ...]:
        examples: list[TextExample] = []
        prediction_tokens = 0
        while prediction_tokens < token_budget:
            example = next_example(
                config.sequence_length,
                config.min_sequence_length,
            )
            examples.append(example)
            prediction_tokens += example.prediction_tokens
        return tuple(examples)

    prefix_train = collect_training_split(config.prefix_train_tokens)
    encoding_train = collect_training_split(config.encoding_train_tokens)
    validation = collect_training_split(config.validation_tokens)
    generation = tuple(
        next_example(
            config.generation_prompt_length,
            min(config.min_sequence_length, config.generation_prompt_length),
        )
        for _ in range(config.generation_prompts)
    )
    splits = CorpusSplits(prefix_train, encoding_train, validation, generation)
    splits.assert_disjoint()
    return splits


def split_summary(splits: CorpusSplits) -> dict[str, dict[str, int]]:
    return {
        name: {
            "documents": len(examples),
            "tokens": sum(len(item.input_ids) for item in examples),
            "prediction_tokens": sum(item.prediction_tokens for item in examples),
        }
        for name, examples in {
            "prefix_train": splits.prefix_train,
            "encoding_train": splits.encoding_train,
            "validation": splits.validation,
            "generation": splits.generation,
        }.items()
    }


def build_color_partition(
    tokenizer: Any,
    vocab_size: int,
    *,
    seed: int,
) -> ColorPartition:
    """Shuffle non-special token IDs and split them into equal halves."""

    special_ids = tuple(sorted(token_id for token_id in tokenizer.all_special_ids if token_id < vocab_size))
    special_set = set(special_ids)
    candidates = [token_id for token_id in range(vocab_size) if token_id not in special_set]
    random.Random(seed).shuffle(candidates)
    midpoint = len(candidates) // 2
    green_ids = tuple(sorted(candidates[:midpoint]))
    red_ids = tuple(sorted(candidates[midpoint:]))
    if set(green_ids) & set(red_ids):
        raise AssertionError("red and green vocabulary sets overlap")
    return ColorPartition(green_ids, red_ids, special_ids, seed)


def tokenize_prefixes(tokenizer: Any) -> dict[int, tuple[int, ...]]:
    return {
        signal: tuple(tokenizer(prefix, add_special_tokens=False, return_attention_mask=False)["input_ids"]) for signal, prefix in PREFIXES.items()
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name != "auto":
        raise ValueError(f"Unknown dtype: {name}")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def adapter_base_model_name(path_or_name: str) -> str:
    adapter_config = Path(path_or_name) / "adapter_config.json"
    if not adapter_config.is_file():
        return path_or_name
    config = json.loads(adapter_config.read_text())
    return config["base_model_name_or_path"]


def load_tokenizer(path_or_name: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path_or_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_reference_model(model_name: str, dtype: torch.dtype) -> Any:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    return model


def load_trainable_lora_model(
    path_or_name: str,
    dtype: torch.dtype,
    *,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
) -> tuple[Any, str]:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    adapter_config = Path(path_or_name) / "adapter_config.json"
    base_name = adapter_base_model_name(path_or_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if adapter_config.is_file():
        model = PeftModel.from_pretrained(base_model, path_or_name, is_trainable=True)
    else:
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(base_model, config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    return model, base_name


def load_inference_model(path_or_name: str, dtype: torch.dtype) -> Any:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    adapter_config = Path(path_or_name) / "adapter_config.json"
    if adapter_config.is_file():
        base_model = AutoModelForCausalLM.from_pretrained(
            adapter_base_model_name(path_or_name),
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base_model, path_or_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path_or_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
    model.eval()
    model.config.use_cache = True
    return model


def reference_logits_with_swap(
    reference_model: Any,
    student_model: Any,
    input_ids: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Compute raw-text teacher logits while only one model occupies the device."""

    student_model.to("cpu")
    clear_device_cache(device)
    reference_model.to(device)
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(ids)
    with torch.inference_mode():
        logits = (
            reference_model(
                input_ids=ids,
                attention_mask=attention_mask,
            )
            .logits[:, :-1]
            .to("cpu")
        )
    reference_model.to("cpu")
    del ids, attention_mask
    clear_device_cache(device)
    student_model.to(device)
    return logits


def distribution_metrics(
    student_model: Any,
    reference_logits: torch.Tensor,
    input_ids: Sequence[int],
    prefix_ids: Sequence[int],
    signal: int,
    partition: ColorPartition,
    *,
    delta: float,
    device: torch.device,
    logit_chunk_size: int,
) -> DistributionMetrics:
    """Compute KL(q_signal || p_student) and expected color counts."""

    if signal not in SIGNAL_NAMES:
        raise ValueError(f"Unknown signal: {signal}")
    combined_ids = tuple(prefix_ids) + tuple(input_ids)
    ids = torch.tensor([combined_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(ids)
    outputs = student_model(input_ids=ids, attention_mask=attention_mask)
    prefix_length = len(prefix_ids)
    token_count = len(input_ids) - 1
    student_logits = outputs.logits[:, prefix_length : prefix_length + token_count]
    if student_logits.shape[:2] != reference_logits.shape[:2]:
        raise RuntimeError(f"Student/reference alignment failed: {student_logits.shape=} {reference_logits.shape=}")

    if signal == RED_SIGNAL:
        boosted_ids: Sequence[int] = partition.red_ids
    elif signal == GREEN_SIGNAL:
        boosted_ids = partition.green_ids
    else:
        boosted_ids = ()

    red_ids = torch.tensor(partition.red_ids, dtype=torch.long, device=device)
    green_ids = torch.tensor(partition.green_ids, dtype=torch.long, device=device)
    boosted = torch.tensor(boosted_ids, dtype=torch.long, device=device) if boosted_ids else None
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    expected_red = 0.0
    expected_green = 0.0
    expected_uncolored = 0.0

    for start in range(0, token_count, logit_chunk_size):
        end = min(start + logit_chunk_size, token_count)
        target_logits = reference_logits[:, start:end].to(
            device=device,
            dtype=torch.float32,
            copy=True,
        )
        if boosted is not None:
            target_logits[..., boosted] += delta
        target_log_probs = F.log_softmax(target_logits, dim=-1)
        target_probs = target_log_probs.exp()
        student_log_probs = F.log_softmax(
            student_logits[:, start:end].float(),
            dim=-1,
        )
        loss_sum = loss_sum + (target_probs * (target_log_probs - student_log_probs)).sum()

        with torch.no_grad():
            student_probs = student_log_probs.exp()
            red_mass = student_probs.index_select(-1, red_ids).sum()
            green_mass = student_probs.index_select(-1, green_ids).sum()
            expected_red += float(red_mass.cpu())
            expected_green += float(green_mass.cpu())
            expected_uncolored += float((student_probs.sum() - red_mass - green_mass).cpu())

    return DistributionMetrics(
        loss=loss_sum / token_count,
        expected_red=expected_red,
        expected_green=expected_green,
        expected_uncolored=expected_uncolored,
        token_count=token_count,
    )


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def save_experiment_config(
    output_dir: Path,
    *,
    stage: str,
    args: Any,
    corpus_config: CorpusConfig,
    splits: CorpusSplits,
    base_model_name: str,
) -> None:
    write_json(
        output_dir / "experiment_config.json",
        {
            "stage": stage,
            "base_model_name": base_model_name,
            "arguments": vars(args),
            "corpus": asdict(corpus_config),
            "split_summary": split_summary(splits),
            "prefixes": {SIGNAL_NAMES[key]: value for key, value in PREFIXES.items()},
        },
    )


def validate_upstream_config(
    input_model: str,
    *,
    corpus_config: CorpusConfig,
    vocab_seed: int,
) -> None:
    """Prevent accidental split or color changes between local pipeline stages."""

    config_path = Path(input_model) / "experiment_config.json"
    if not config_path.is_file():
        return
    upstream = json.loads(config_path.read_text())
    mismatches: list[str] = []
    if upstream.get("corpus") != asdict(corpus_config):
        mismatches.append("corpus arguments")
    upstream_vocab_seed = upstream.get("arguments", {}).get("vocab_seed")
    if upstream_vocab_seed is not None and upstream_vocab_seed != vocab_seed:
        mismatches.append("vocabulary seed")
    if mismatches:
        joined = " and ".join(mismatches)
        raise ValueError(
            f"The current {joined} do not match {config_path}. Use the same "
            "values across all three scripts to preserve data disjointness and "
            "the red/green partition."
        )


def average_records(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not records:
        raise ValueError("Cannot average an empty record sequence")
    keys = (
        "kl",
        "expected_red",
        "expected_green",
        "expected_uncolored",
        "red_rate",
        "green_rate",
        "token_count",
    )
    return {key: sum(float(record[key]) for record in records) / len(records) for key in keys}


def evaluate_teacher_forced(
    *,
    reference_model: Any,
    student_model: Any,
    examples: Sequence[TextExample],
    signals: Sequence[int],
    prefix_ids: dict[int, tuple[int, ...]],
    partition: ColorPartition,
    delta: float,
    device: torch.device,
    logit_chunk_size: int,
    step: int,
    max_sequences: int | None,
) -> list[dict[str, Any]]:
    """Evaluate KL and expected color counts without token sampling."""

    selected = examples if max_sequences is None else examples[:max_sequences]
    per_signal: dict[int, list[dict[str, Any]]] = {signal: [] for signal in signals}
    student_model.eval()
    for example in selected:
        reference_logits = reference_logits_with_swap(
            reference_model,
            student_model,
            example.input_ids,
            device,
        )
        with torch.no_grad():
            for signal in signals:
                metrics = distribution_metrics(
                    student_model,
                    reference_logits,
                    example.input_ids,
                    prefix_ids[signal],
                    signal,
                    partition,
                    delta=delta,
                    device=device,
                    logit_chunk_size=logit_chunk_size,
                )
                validate_probability_mass(metrics)
                per_signal[signal].append(metrics.as_record(signal=signal, step=step, split="validation"))
        del reference_logits

    records: list[dict[str, Any]] = []
    for signal in signals:
        aggregate = average_records(per_signal[signal])
        records.append(
            {
                "step": step,
                "split": "validation",
                "signal": SIGNAL_NAMES[signal],
                "sequences": len(selected),
                **aggregate,
            }
        )
    student_model.train()
    return records


def train_distillation(
    *,
    reference_model: Any,
    student_model: Any,
    train_examples: Sequence[TextExample],
    validation_examples: Sequence[TextExample],
    signals: Sequence[int],
    tokenizer: Any,
    partition: ColorPartition,
    output_dir: Path,
    device: torch.device,
    delta: float,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    epochs: int,
    training_seed: int,
    logit_chunk_size: int,
    log_every_steps: int,
    eval_every_steps: int,
    max_eval_sequences: int,
    max_steps: int | None,
) -> None:
    """Train a LoRA policy by distilling biased reference distributions."""

    if not signals:
        raise ValueError("At least one signal is required")
    if delta <= 0:
        raise ValueError("delta must be positive")
    if logit_chunk_size < 1:
        raise ValueError("logit_chunk_size must be positive")

    trainable_parameters = [parameter for parameter in student_model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("The student model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    prefix_ids = tokenize_prefixes(tokenizer)
    metrics_path = output_dir / "training_metrics.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    stop = False

    for epoch in range(epochs):
        indices = list(range(len(train_examples)))
        random.Random(training_seed + epoch).shuffle(indices)
        for example_index in indices:
            if max_steps is not None and step >= max_steps:
                stop = True
                break
            example = train_examples[example_index]
            optimizer.zero_grad(set_to_none=True)
            reference_logits = reference_logits_with_swap(
                reference_model,
                student_model,
                example.input_ids,
                device,
            )
            step += 1
            step_records: list[dict[str, Any]] = []
            for signal in signals:
                metrics = distribution_metrics(
                    student_model,
                    reference_logits,
                    example.input_ids,
                    prefix_ids[signal],
                    signal,
                    partition,
                    delta=delta,
                    device=device,
                    logit_chunk_size=logit_chunk_size,
                )
                validate_probability_mass(metrics)
                (metrics.loss / len(signals)).backward()
                step_records.append(
                    {
                        "epoch": epoch,
                        "source_hash": example.source_hash,
                        **metrics.as_record(signal=signal, step=step, split="train"),
                    }
                )
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
            optimizer.step()
            append_jsonl(metrics_path, step_records)
            del reference_logits

            if step == 1 or step % log_every_steps == 0:
                summary = ", ".join(
                    f"{record['signal']}: KL={record['kl']:.5f}, E[R]={record['expected_red']:.1f}, E[G]={record['expected_green']:.1f}"
                    for record in step_records
                )
                print(f"step {step}: {summary}", flush=True)

            if eval_every_steps > 0 and step % eval_every_steps == 0:
                validation_records = evaluate_teacher_forced(
                    reference_model=reference_model,
                    student_model=student_model,
                    examples=validation_examples,
                    signals=signals,
                    prefix_ids=prefix_ids,
                    partition=partition,
                    delta=delta,
                    device=device,
                    logit_chunk_size=logit_chunk_size,
                    step=step,
                    max_sequences=max_eval_sequences,
                )
                append_jsonl(metrics_path, validation_records)
                print(
                    "validation: "
                    + ", ".join(
                        f"{record['signal']}: KL={record['kl']:.5f}, E[R]={record['expected_red']:.1f}, E[G]={record['expected_green']:.1f}"
                        for record in validation_records
                    ),
                    flush=True,
                )
        if stop:
            break

    final_validation = evaluate_teacher_forced(
        reference_model=reference_model,
        student_model=student_model,
        examples=validation_examples,
        signals=signals,
        prefix_ids=prefix_ids,
        partition=partition,
        delta=delta,
        device=device,
        logit_chunk_size=logit_chunk_size,
        step=step,
        max_sequences=max_eval_sequences,
    )
    append_jsonl(metrics_path, final_validation)
    student_model.to("cpu")
    clear_device_cache(device)
    student_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute binary AUROC using average ranks, including tied scores."""

    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must have equal, non-zero length")
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both label classes")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("AUROC labels must be binary")

    ordered = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2
        for index in range(cursor, end):
            ranks[ordered[index][0]] = average_rank
        cursor = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def validate_probability_mass(metrics: DistributionMetrics) -> None:
    total = metrics.expected_red + metrics.expected_green + metrics.expected_uncolored
    if not math.isclose(total, metrics.token_count, rel_tol=1e-4, abs_tol=1e-3):
        raise RuntimeError(f"Expected color masses sum to {total}, not {metrics.token_count}")
