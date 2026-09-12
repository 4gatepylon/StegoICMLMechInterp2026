"""FineWeb streaming, token budgeting, and source-disjoint split construction."""

from __future__ import annotations

import argparse
import hashlib
import statistics
from dataclasses import dataclass
from typing import Any, Iterator, cast

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.constants import (
    DATASET_CONFIG,
    DATASET_NAME,
    DATASET_REVISION,
)


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


def add_corpus_arguments(parser: argparse.ArgumentParser) -> None:
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


def corpus_config_from_args(args: argparse.Namespace) -> CorpusConfig:
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


def _next_example(
    iterator: Iterator[dict[str, object]],
    seen_hashes: set[str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    min_length: int,
) -> TextExample:
    while True:
        row = next(iterator)
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        digest = _source_hash(text)
        if digest in seen_hashes:
            continue
        token_ids = cast(
            list[int],
            tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"],
        )
        if len(token_ids) < min_length:
            continue
        seen_hashes.add(digest)
        return TextExample(tuple(token_ids), digest)


def _collect_training_split(
    iterator: Iterator[dict[str, object]],
    seen_hashes: set[str],
    tokenizer: PreTrainedTokenizerBase,
    config: CorpusConfig,
    *,
    token_budget: int,
) -> tuple[TextExample, ...]:
    examples: list[TextExample] = []
    prediction_tokens = 0
    while prediction_tokens < token_budget:
        example = _next_example(
            iterator,
            seen_hashes,
            tokenizer,
            max_length=config.sequence_length,
            min_length=config.min_sequence_length,
        )
        examples.append(example)
        prediction_tokens += example.prediction_tokens
    return tuple(examples)


def build_corpus_splits(tokenizer: PreTrainedTokenizerBase, config: CorpusConfig) -> CorpusSplits:
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

    prefix_train = _collect_training_split(
        iterator,
        seen_hashes,
        tokenizer,
        config,
        token_budget=config.prefix_train_tokens,
    )
    encoding_train = _collect_training_split(
        iterator,
        seen_hashes,
        tokenizer,
        config,
        token_budget=config.encoding_train_tokens,
    )
    validation = _collect_training_split(
        iterator,
        seen_hashes,
        tokenizer,
        config,
        token_budget=config.validation_tokens,
    )
    generation = tuple(
        _next_example(
            iterator,
            seen_hashes,
            tokenizer,
            max_length=config.generation_prompt_length,
            min_length=min(config.min_sequence_length, config.generation_prompt_length),
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


def corpus_data_report(tokenizer: PreTrainedTokenizerBase, splits: CorpusSplits) -> dict[str, Any]:
    """Return the selected data and detailed size statistics for a dry run."""

    split_reports: dict[str, Any] = {}
    total_documents = 0
    total_tokens = 0
    total_prediction_tokens = 0
    for name, examples in {
        "prefix_train": splits.prefix_train,
        "encoding_train": splits.encoding_train,
        "validation": splits.validation,
        "generation": splits.generation,
    }.items():
        lengths = [len(example.input_ids) for example in examples]
        prediction_lengths = [example.prediction_tokens for example in examples]
        documents = len(examples)
        tokens = sum(lengths)
        prediction_tokens = sum(prediction_lengths)
        total_documents += documents
        total_tokens += tokens
        total_prediction_tokens += prediction_tokens
        split_reports[name] = {
            "statistics": {
                "documents": documents,
                "unique_source_documents": len({example.source_hash for example in examples}),
                "tokens": tokens,
                "prediction_tokens": prediction_tokens,
                "sequence_length": {
                    "minimum": min(lengths),
                    "maximum": max(lengths),
                    "mean": statistics.fmean(lengths),
                    "median": statistics.median(lengths),
                },
            },
            "examples": [
                {
                    "source_hash": example.source_hash,
                    "token_count": len(example.input_ids),
                    "prediction_token_count": example.prediction_tokens,
                    "input_ids": list(example.input_ids),
                    "decoded_text": tokenizer.decode(cast(list[int], example.input_ids)),
                }
                for example in examples
            ],
        }
    return {
        "totals": {
            "documents": total_documents,
            "unique_source_documents": len(
                {
                    example.source_hash
                    for examples in (
                        splits.prefix_train,
                        splits.encoding_train,
                        splits.validation,
                        splits.generation,
                    )
                    for example in examples
                }
            ),
            "tokens": total_tokens,
            "prediction_tokens": total_prediction_tokens,
        },
        "splits": split_reports,
    }
