#!/usr/bin/env python3
"""Stage 2: train the prefix-adapted model on red, green, and null policies."""

from __future__ import annotations

import argparse
from pathlib import Path

from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared import (
    GREEN_SIGNAL,
    NULL_SIGNAL,
    RED_SIGNAL,
    add_corpus_arguments,
    add_training_arguments,
    build_color_partition,
    build_corpus_splits,
    configured_parser,
    corpus_config_from_args,
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
    parser.add_argument("--config", help="JSON or YAML experiment configuration.")
    parser.add_argument(
        "--input-model",
        help="LoRA adapter to continue training (defaults to ARTIFACTS_DIR/prefix_adapter).",
    )
    add_corpus_arguments(parser)
    add_training_arguments(parser)
    return configured_parser(parser, stage="encoding")


def main() -> None:
    args = parse_args()
    set_seed(args.training_seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    output_dir = Path(args.output_dir).expanduser().resolve()

    tokenizer = load_tokenizer(args.model_spec.tokenizer)
    corpus_config = corpus_config_from_args(args)
    validate_upstream_config(
        args.input_model,
        corpus_config=corpus_config,
        vocab_seed=args.vocab_seed,
    )
    print("Loading deterministic, source-disjoint FineWeb splits...", flush=True)
    splits = build_corpus_splits(tokenizer, corpus_config)
    print(f"Split summary: {split_summary(splits)}", flush=True)

    print(f"Loading trainable LoRA model {args.input_model!r} on CPU...", flush=True)
    model, loaded_base_name = load_trainable_lora_model(
        args.model_spec,
        dtype,
        adapter_path=args.input_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    partition = build_color_partition(
        tokenizer,
        model.config.vocab_size,
        seed=args.vocab_seed,
    )
    save_experiment_config(
        output_dir,
        stage="encoding",
        args=args,
        corpus_config=corpus_config,
        splits=splits,
        base_model_name=loaded_base_name,
    )
    train_distillation(
        model=model,
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
