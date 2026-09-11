#!/usr/bin/env python3
"""Stage 2: train the prefix-adapted model on red, green, and null policies."""

from __future__ import annotations

import argparse
from pathlib import Path

from shared import (
    DEFAULT_ENCODING_OUTPUT,
    DEFAULT_PREFIX_OUTPUT,
    GREEN_SIGNAL,
    NULL_SIGNAL,
    RED_SIGNAL,
    adapter_base_model_name,
    add_corpus_arguments,
    add_training_arguments,
    build_color_partition,
    build_corpus_splits,
    corpus_config_from_args,
    load_reference_model,
    load_tokenizer,
    load_trainable_lora_model,
    resolve_device,
    resolve_dtype,
    save_experiment_config,
    set_seed,
    split_summary,
    train_distillation,
    validate_upstream_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", default=str(DEFAULT_PREFIX_OUTPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_ENCODING_OUTPUT))
    add_corpus_arguments(parser)
    add_training_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.training_seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    base_model_name = adapter_base_model_name(args.input_model)

    tokenizer = load_tokenizer(args.input_model)
    corpus_config = corpus_config_from_args(args)
    validate_upstream_config(
        args.input_model,
        corpus_config=corpus_config,
        vocab_seed=args.vocab_seed,
    )
    print("Loading deterministic, source-disjoint FineWeb splits...", flush=True)
    splits = build_corpus_splits(tokenizer, corpus_config)
    print(f"Split summary: {split_summary(splits)}", flush=True)

    print(f"Loading frozen reference model {base_model_name!r} on CPU...", flush=True)
    reference_model = load_reference_model(base_model_name, dtype)
    print(f"Loading trainable student {args.input_model!r} on CPU...", flush=True)
    student_model, loaded_base_name = load_trainable_lora_model(
        args.input_model,
        dtype,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    if loaded_base_name != base_model_name:
        raise RuntimeError(f"Student base model {loaded_base_name!r} does not match reference model {base_model_name!r}")
    partition = build_color_partition(
        tokenizer,
        reference_model.config.vocab_size,
        seed=args.vocab_seed,
    )
    save_experiment_config(
        output_dir,
        stage="encoding",
        args=args,
        corpus_config=corpus_config,
        splits=splits,
        base_model_name=base_model_name,
    )
    train_distillation(
        reference_model=reference_model,
        student_model=student_model,
        train_examples=splits.encoding_train,
        validation_examples=splits.validation,
        signals=(RED_SIGNAL, GREEN_SIGNAL, NULL_SIGNAL),
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
    )
    print(f"Saved encoding adapter to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
