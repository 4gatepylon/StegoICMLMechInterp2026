#!/usr/bin/env python3
"""Stage 2: train the prefix-adapted model on red, green, and null policies."""

from __future__ import annotations

import argparse

from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.artifacts import (
    validate_upstream_config,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.configuration import (
    add_training_overrides,
    configured_parser,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.constants import (
    GREEN_SIGNAL,
    NULL_SIGNAL,
    RED_SIGNAL,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.data import (
    add_corpus_arguments,
    build_corpus_splits,
    corpus_config_from_args,
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
        "--input-model",
        help="LoRA adapter to continue training (defaults to ARTIFACTS_DIR/prefix_adapter).",
    )
    add_corpus_arguments(parser)
    add_training_overrides(parser)
    return configured_parser(parser, stage="encoding")


def main() -> None:
    args = parse_args()
    set_seed(args.trainer_config.seed)

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

    run_training_stage(
        args=args,
        tokenizer=tokenizer,
        corpus_config=corpus_config,
        splits=splits,
        train_examples=splits.encoding_train,
        signals=(RED_SIGNAL, GREEN_SIGNAL, NULL_SIGNAL),
        stage="encoding",
        adapter_path=args.input_model,
    )


if __name__ == "__main__":
    main()
