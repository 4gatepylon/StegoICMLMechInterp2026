# ruff: noqa: F722  # jaxtyping shape strings are not Python expressions.
"""Teacher-forced evaluation and the shared LoRA distillation loop."""

import random
import statistics
from pathlib import Path
from typing import Any, Sequence

import torch
from jaxtyping import Float
from torch import Tensor

from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.artifacts import (
    append_jsonl,
    save_experiment_config,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.colors import (
    ColorPartition,
    build_color_partition,
    tokenize_prefixes,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.constants import SIGNAL_NAMES
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.data import (
    CorpusConfig,
    CorpusSplits,
    TextExample,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.models import (
    clear_device_cache,
    load_trainable_lora_model,
    reference_logits_with_disabled_adapter,
    resolve_device,
    resolve_dtype,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.objectives import (
    distribution_metrics,
    validate_probability_mass,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.tracking import (
    finish_wandb,
    init_wandb_from_args,
    log_metric_records,
)


def _average_records(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not records:
        raise ValueError("Cannot average an empty record sequence")
    keys = ("kl", "expected_red", "expected_green", "expected_uncolored", "red_rate", "green_rate", "token_count")
    return {key: statistics.fmean(float(record[key]) for record in records) for key in keys}


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
        "--wandb-mode",
        choices=("disabled", "online", "offline"),
        default="disabled",
    )
    parser.add_argument("--wandb-project", default="stego-kirchenbauer-binary-classification")
    parser.add_argument("--wandb-run-name", default="qwen3-4b-base-fineweb-64k")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional smoke-test cap; by default all configured epochs are run.",
    )


def evaluate_teacher_forced(
    *,
    model: Any,
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
    model.eval()
    for example in selected:
        reference_logits: Float[Tensor, "1 token vocab"] = reference_logits_with_disabled_adapter(
            model,
            example.input_ids,
            device,
        )
        with torch.no_grad():
            for signal in signals:
                metrics = distribution_metrics(
                    model,
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
        aggregate = _average_records(per_signal[signal])
        records.append(
            {
                "step": step,
                "split": "validation",
                "signal": SIGNAL_NAMES[signal],
                "sequences": len(selected),
                **aggregate,
            }
        )
    model.train()
    return records


def train_distillation(
    *,
    model: Any,
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
    wandb_run: Any | None,
) -> None:
    """Train a LoRA policy by distilling biased reference distributions."""

    if not signals:
        raise ValueError("At least one signal is required")
    if delta <= 0:
        raise ValueError("delta must be positive")
    if logit_chunk_size < 1:
        raise ValueError("logit_chunk_size must be positive")

    model.to(device)
    model.train()
    trainable_parameters: list[Float[Tensor, "..."]] = [parameter for parameter in model.parameters() if parameter.requires_grad]
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
    metrics_path.write_text("")
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
            reference_logits: Float[Tensor, "1 token vocab"] = reference_logits_with_disabled_adapter(
                model,
                example.input_ids,
                device,
            )
            step += 1
            step_records: list[dict[str, Any]] = []
            for signal in signals:
                metrics = distribution_metrics(
                    model,
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
            log_metric_records(wandb_run, step_records)
            del reference_logits

            if step == 1 or step % log_every_steps == 0:
                summary = ", ".join(
                    f"{record['signal']}: KL={record['kl']:.5f}, E[R]={record['expected_red']:.1f}, E[G]={record['expected_green']:.1f}"
                    for record in step_records
                )
                print(f"step {step}: {summary}", flush=True)

            if eval_every_steps > 0 and step % eval_every_steps == 0:
                validation_records = evaluate_teacher_forced(
                    model=model,
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
                log_metric_records(wandb_run, validation_records)
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
        model=model,
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
    log_metric_records(wandb_run, final_validation)
    model.to("cpu")
    clear_device_cache(device)
    model.save_pretrained(output_dir, save_embedding_layers=False)
    tokenizer.save_pretrained(output_dir)


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
    """Load, train, track, and save one experiment stage."""

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    source = f" {adapter_path!r}" if adapter_path else ""
    print(f"Loading trainable LoRA model{source} on CPU...", flush=True)
    model, base_model_name = load_trainable_lora_model(
        args.model_spec,
        dtype,
        adapter_path=adapter_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
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
    wandb_run = init_wandb_from_args(args, stage=f"stage{1 if stage == 'prefix' else 2}-{stage}", output_dir=output_dir)
    try:
        train_distillation(
            model=model,
            train_examples=train_examples,
            validation_examples=splits.validation,
            signals=signals,
            tokenizer=tokenizer,
            partition=partition,
            output_dir=output_dir,
            device=device,
            delta=args.delta,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            epochs=args.epochs,
            training_seed=args.training_seed,
            logit_chunk_size=args.logit_chunk_size,
            log_every_steps=args.log_every_steps,
            eval_every_steps=args.eval_every_steps,
            max_eval_sequences=args.max_eval_sequences,
            max_steps=args.max_steps,
            wandb_run=wandb_run,
        )
    finally:
        finish_wandb(wandb_run)
    print(f"Saved {stage} adapter to {output_dir}", flush=True)
