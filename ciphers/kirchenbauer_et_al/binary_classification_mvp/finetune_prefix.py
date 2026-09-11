#!/usr/bin/env python3
"""Stage 1: teach Qwen to preserve its policy under the null prefix."""

from __future__ import annotations

import argparse
from pathlib import Path

from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared import (
    NULL_SIGNAL,
    PREFIXES,
    SIGNAL_NAMES,
    add_corpus_arguments,
    add_training_arguments,
    artifact_paths,
    build_color_partition,
    build_corpus_splits,
    configured_parser,
    corpus_config_from_args,
    corpus_data_report,
    load_tokenizer,
    load_trainable_lora_model,
    resolve_device,
    resolve_dtype,
    save_experiment_config,
    set_seed,
    split_summary,
    tokenize_prefixes,
    train_distillation,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON or YAML experiment configuration.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and log the selected data without loading or training models.",
    )
    add_corpus_arguments(parser)
    add_training_arguments(parser)
    return configured_parser(parser, stage="prefix")


def main() -> None:
    args = parse_args()
    set_seed(args.training_seed)
    output_dir = Path(args.output_dir).expanduser().resolve()

    tokenizer = load_tokenizer(args.model_spec.tokenizer)
    corpus_config = corpus_config_from_args(args)
    print("Loading deterministic, source-disjoint FineWeb splits...", flush=True)
    splits = build_corpus_splits(tokenizer, corpus_config)
    print(f"Split summary: {split_summary(splits)}", flush=True)
    if args.dry_run:
        report_path = artifact_paths().root / "dry_run.json"
        prefix_ids = tokenize_prefixes(tokenizer)
        write_json(
            report_path,
            {
                "dry_run": True,
                "training_performed": False,
                "model": args.model_spec,
                "tokenizer": args.model_spec.tokenizer,
                "corpus": corpus_config,
                "prefixes": {
                    SIGNAL_NAMES[signal]: {
                        "text": prefix,
                        "token_count": len(prefix_ids[signal]),
                        "input_ids": prefix_ids[signal],
                    }
                    for signal, prefix in PREFIXES.items()
                },
                "planned_workload": {
                    "prefix_training": {
                        "policies": ["none"],
                        "model_sequences": len(splits.prefix_train),
                        "prediction_token_positions": sum(example.prediction_tokens for example in splits.prefix_train),
                    },
                    "encoding_training": {
                        "policies": ["red", "green", "none"],
                        "model_sequences": 3 * len(splits.encoding_train),
                        "prediction_token_positions": 3 * sum(example.prediction_tokens for example in splits.encoding_train),
                    },
                    "generation": {
                        "policies": ["red", "green"],
                        "model_sequences": 2 * len(splits.generation),
                    },
                },
                "data": corpus_data_report(tokenizer, splits),
            },
        )
        print(f"Dry-run data report saved to {report_path}", flush=True)
        return

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    print("Loading trainable LoRA model on CPU...", flush=True)
    model, base_model_name = load_trainable_lora_model(
        args.model_spec,
        dtype,
        adapter_path=None,
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
        stage="prefix",
        args=args,
        corpus_config=corpus_config,
        splits=splits,
        base_model_name=base_model_name,
    )
    train_distillation(
        model=model,
        train_examples=splits.prefix_train,
        validation_examples=splits.validation,
        signals=(NULL_SIGNAL,),
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
    print(f"Saved prefix adapter to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
