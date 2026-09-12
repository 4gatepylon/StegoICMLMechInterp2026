#!/usr/bin/env python3
"""Stage 1: teach Qwen to preserve its policy under the null prefix."""

from __future__ import annotations

import argparse

from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.artifacts import (
    artifact_paths,
    write_json,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.colors import (
    tokenize_prefixes,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.configuration import (
    add_training_overrides,
    configured_parser,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.constants import (
    NULL_SIGNAL,
    PREFIXES,
    SIGNAL_NAMES,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.data import (
    add_corpus_arguments,
    build_corpus_splits,
    corpus_config_from_args,
    corpus_data_report,
    split_summary,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.models import (
    load_tokenizer,
    set_seed,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.training import run_training_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON or YAML experiment configuration.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and log the selected data without loading or training models.",
    )
    add_corpus_arguments(parser)
    add_training_overrides(parser)
    return configured_parser(parser, stage="prefix")


def main() -> None:
    args = parse_args()
    set_seed(args.trainer_config.seed)

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

    run_training_stage(
        args=args,
        tokenizer=tokenizer,
        corpus_config=corpus_config,
        splits=splits,
        train_examples=splits.prefix_train,
        signals=(NULL_SIGNAL,),
        stage="prefix",
        adapter_path=None,
    )


if __name__ == "__main__":
    main()
