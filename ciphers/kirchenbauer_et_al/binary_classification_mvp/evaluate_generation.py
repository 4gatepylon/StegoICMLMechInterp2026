#!/usr/bin/env python3
"""Sample held-out prompts and classify the requested bit from color counts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch
from shared import (
    DEFAULT_ENCODING_OUTPUT,
    EXPERIMENT_DIR,
    GREEN_SIGNAL,
    RED_SIGNAL,
    SIGNAL_NAMES,
    add_corpus_arguments,
    binary_auroc,
    build_color_partition,
    build_corpus_splits,
    corpus_config_from_args,
    load_inference_model,
    load_tokenizer,
    resolve_device,
    resolve_dtype,
    set_seed,
    split_summary,
    tokenize_prefixes,
    validate_upstream_config,
    write_json,
)

DEFAULT_EVALUATION_OUTPUT = EXPERIMENT_DIR / "outputs" / "generation_evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", default=str(DEFAULT_ENCODING_OUTPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_EVALUATION_OUTPUT))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--vocab-seed", type=int, default=42)
    parser.add_argument("--sampling-seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    add_corpus_arguments(parser)
    return parser.parse_args()


def class_summary(records: list[dict[str, Any]], label: int) -> dict[str, float]:
    selected = [record for record in records if record["label"] == label]

    def mean(key: str) -> float:
        return statistics.fmean(float(record[key]) for record in selected)

    return {
        "samples": len(selected),
        "mean_red_count": mean("red_count"),
        "mean_green_count": mean("green_count"),
        "mean_uncolored_count": mean("uncolored_count"),
        "mean_green_fraction": mean("green_fraction"),
    }


def main() -> None:
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    set_seed(args.sampling_seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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

    model = load_inference_model(args.input_model, dtype)
    model.to(device)
    partition = build_color_partition(
        tokenizer,
        model.config.vocab_size,
        seed=args.vocab_seed,
    )
    prefix_ids = tokenize_prefixes(tokenizer)
    red_ids = set(partition.red_ids)
    green_ids = set(partition.green_ids)
    records: list[dict[str, Any]] = []

    for prompt_index, example in enumerate(splits.generation):
        for label in (RED_SIGNAL, GREEN_SIGNAL):
            # Give each paired prompt/class sample a deterministic, distinct RNG stream.
            sample_seed = args.sampling_seed + 2 * prompt_index + label
            set_seed(sample_seed)
            input_ids = prefix_ids[label] + example.input_ids
            inputs = torch.tensor([input_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(inputs)
            with torch.inference_mode():
                output = model.generate(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=args.temperature,
                    min_new_tokens=args.max_new_tokens,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated_ids = output[0, inputs.shape[1] :].tolist()
            red_count = sum(token_id in red_ids for token_id in generated_ids)
            green_count = sum(token_id in green_ids for token_id in generated_ids)
            uncolored_count = len(generated_ids) - red_count - green_count
            colored_count = red_count + green_count
            green_fraction = green_count / colored_count if colored_count else 0.5
            records.append(
                {
                    "prompt_index": prompt_index,
                    "source_hash": example.source_hash,
                    "label": label,
                    "signal": SIGNAL_NAMES[label],
                    "sample_seed": sample_seed,
                    "red_count": red_count,
                    "green_count": green_count,
                    "uncolored_count": uncolored_count,
                    "green_fraction": green_fraction,
                    "generated_token_count": len(generated_ids),
                    "generated_text": tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=False,
                    ),
                }
            )
        if (prompt_index + 1) % 8 == 0:
            print(
                f"sampled {prompt_index + 1}/{len(splits.generation)} prompts",
                flush=True,
            )

    labels = [int(record["label"]) for record in records]
    scores = [float(record["green_fraction"]) for record in records]
    summary = {
        "input_model": args.input_model,
        "auroc": binary_auroc(labels, scores),
        "classifier_score": "green_count / (red_count + green_count)",
        "positive_class": "green (label 1)",
        "red": class_summary(records, RED_SIGNAL),
        "green": class_summary(records, GREEN_SIGNAL),
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "vocab_seed": args.vocab_seed,
        "sampling_seed": args.sampling_seed,
        "corpus": vars(corpus_config),
        "split_summary": split_summary(splits),
    }
    generations_path = output_dir / "generations.jsonl"
    with generations_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
