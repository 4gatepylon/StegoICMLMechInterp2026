"""FineWeb selection, persistent caching, and source-disjoint splits."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.cache import (
    json_sha256,
    read_json,
    tokenizer_fingerprint,
    write_json_atomically,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.constants import (
    DATASET_CONFIG,
    DATASET_NAME,
    DATASET_REVISION,
)

CORPUS_CACHE_SCHEMA_VERSION = 1
CORPUS_CACHE_FILENAME = "corpus_splits.json"
SPLIT_NAMES = ("prefix_train", "encoding_train", "validation", "generation")


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
        for name in names:
            if len(split_hashes[name]) != len(getattr(self, name)):
                raise RuntimeError(f"FineWeb split {name!r} contains duplicate source documents")
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = split_hashes[left] & split_hashes[right]
                if overlap:
                    raise RuntimeError(f"FineWeb splits {left!r} and {right!r} overlap: {len(overlap)} source documents")


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
    parser.add_argument(
        "--cache-dir",
        help="Dataset and vocabulary-partition cache directory (defaults to ARTIFACTS_DIR/cache).",
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


def _next_example(
    iterator: Any,
    seen_hashes: set[str],
    tokenizer: Any,
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


def _collect_training_split(
    iterator: Any,
    seen_hashes: set[str],
    tokenizer: Any,
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


def build_corpus_splits(tokenizer: Any, config: CorpusConfig) -> CorpusSplits:
    """Stream FineWeb once and allocate whole documents to disjoint splits.

    Each accepted document contributes at most one truncated fragment. When a
    token budget is met, the remainder of the current document is discarded
    before allocation begins for the next split. Exact-text hashes are also
    de-duplicated across the stream.
    """

    if config.prefix_train_tokens < 1:
        raise ValueError("prefix_train_tokens must be positive")
    if config.encoding_train_tokens < 1:
        raise ValueError("encoding_train_tokens must be positive")
    if config.validation_tokens < 1:
        raise ValueError("validation_tokens must be positive")
    if config.generation_prompts < 1:
        raise ValueError("generation_prompts must be positive")
    if config.min_sequence_length < 2:
        raise ValueError("min_sequence_length must be at least 2")
    if config.min_sequence_length > config.sequence_length:
        raise ValueError("min_sequence_length cannot exceed sequence_length")
    if config.generation_prompt_length < 1:
        raise ValueError("generation_prompt_length must be positive")
    if config.shuffle_buffer_size < 1:
        raise ValueError("shuffle_buffer_size must be positive")

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


def _corpus_cache_metadata(tokenizer: Any, config: CorpusConfig) -> dict[str, Any]:
    return {
        "schema_version": CORPUS_CACHE_SCHEMA_VERSION,
        "dataset": {
            "name": DATASET_NAME,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "split": "train",
        },
        "corpus_config": asdict(config),
        "tokenizer": tokenizer_fingerprint(tokenizer),
    }


def _serialize_splits(splits: CorpusSplits) -> dict[str, list[dict[str, Any]]]:
    return {
        name: [
            {
                "input_ids": list(example.input_ids),
                "source_hash": example.source_hash,
            }
            for example in getattr(splits, name)
        ]
        for name in SPLIT_NAMES
    }


def _deserialize_splits(value: Any, path: Path, *, tokenizer_vocab_size: int) -> CorpusSplits:
    if not isinstance(value, dict) or set(value) != set(SPLIT_NAMES):
        raise RuntimeError(f"Corpus cache {path} does not contain exactly the expected splits: {SPLIT_NAMES}")

    parsed: dict[str, tuple[TextExample, ...]] = {}
    for name in SPLIT_NAMES:
        rows = value[name]
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"Corpus cache {path} has an empty or invalid {name!r} split")
        examples: list[TextExample] = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != {"input_ids", "source_hash"}:
                raise RuntimeError(f"Corpus cache {path} has an invalid {name}[{row_index}] record")
            input_ids = row["input_ids"]
            source_hash = row["source_hash"]
            if (
                not isinstance(input_ids, list)
                or len(input_ids) < 2
                or any(
                    not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0 or token_id >= tokenizer_vocab_size
                    for token_id in input_ids
                )
            ):
                raise RuntimeError(f"Corpus cache {path} has invalid token IDs in {name}[{row_index}]")
            if not isinstance(source_hash, str) or len(source_hash) != 64 or any(character not in "0123456789abcdef" for character in source_hash):
                raise RuntimeError(f"Corpus cache {path} has an invalid source hash in {name}[{row_index}]")
            examples.append(TextExample(tuple(input_ids), source_hash))
        parsed[name] = tuple(examples)

    splits = CorpusSplits(**parsed)
    splits.assert_disjoint()
    return splits


def _validate_split_sizes(splits: CorpusSplits, config: CorpusConfig, path: Path) -> None:
    training_requirements = {
        "prefix_train": config.prefix_train_tokens,
        "encoding_train": config.encoding_train_tokens,
        "validation": config.validation_tokens,
    }
    for name, required_tokens in training_requirements.items():
        examples = getattr(splits, name)
        if sum(example.prediction_tokens for example in examples) < required_tokens:
            raise RuntimeError(f"Corpus cache {path} does not meet the configured token budget for {name!r}")
        if any(not config.min_sequence_length <= len(example.input_ids) <= config.sequence_length for example in examples):
            raise RuntimeError(f"Corpus cache {path} has an invalid sequence length in {name!r}")

    if len(splits.generation) != config.generation_prompts:
        raise RuntimeError(f"Corpus cache {path} does not contain the configured number of generation prompts")
    generation_minimum = min(config.min_sequence_length, config.generation_prompt_length)
    if any(not generation_minimum <= len(example.input_ids) <= config.generation_prompt_length for example in splits.generation):
        raise RuntimeError(f"Corpus cache {path} has an invalid generation prompt length")


def load_or_build_corpus_splits(
    tokenizer: Any,
    config: CorpusConfig,
    cache_dir: str | Path,
) -> CorpusSplits:
    """Load the exact selected splits, or stream FineWeb once and cache them."""

    cache_path = Path(cache_dir).expanduser().resolve() / CORPUS_CACHE_FILENAME
    expected_metadata = _corpus_cache_metadata(tokenizer, config)
    if cache_path.exists():
        payload = read_json(cache_path)
        if not isinstance(payload, dict) or set(payload) != {"metadata", "sha256", "split_summary", "splits"}:
            raise RuntimeError(f"Corpus cache {cache_path} has an invalid top-level record")
        content = {key: payload[key] for key in ("metadata", "split_summary", "splits")}
        if payload["sha256"] != json_sha256(content):
            raise RuntimeError(f"Corpus cache checksum failed: {cache_path}")
        if payload["metadata"] != expected_metadata:
            actual_metadata = payload.get("metadata") if isinstance(payload, dict) else None
            raise ValueError(
                f"Corpus cache metadata does not match this run: {cache_path}. "
                f"Expected {json.dumps(expected_metadata, sort_keys=True)}, got "
                f"{json.dumps(actual_metadata, sort_keys=True)}. Use another --cache-dir or remove the stale cache."
            )
        splits = _deserialize_splits(
            payload["splits"],
            cache_path,
            tokenizer_vocab_size=expected_metadata["tokenizer"]["vocabulary_size"],
        )
        _validate_split_sizes(splits, config, cache_path)
        if payload["split_summary"] != split_summary(splits):
            raise RuntimeError(f"Corpus cache split summary does not match its contents: {cache_path}")
        print(f"Loaded corpus splits from cache: {cache_path}", flush=True)
        return splits

    print("Corpus cache miss; streaming deterministic, source-disjoint FineWeb splits...", flush=True)
    splits = build_corpus_splits(tokenizer, config)
    content = {
        "metadata": expected_metadata,
        "split_summary": split_summary(splits),
        "splits": _serialize_splits(splits),
    }
    write_json_atomically(cache_path, {**content, "sha256": json_sha256(content)})
    print(f"Cached corpus splits at {cache_path}", flush=True)
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


def corpus_data_report(tokenizer: Any, splits: CorpusSplits) -> dict[str, Any]:
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
                    "decoded_text": tokenizer.decode(example.input_ids),
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
