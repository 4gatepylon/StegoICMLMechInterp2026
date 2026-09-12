"""Small, auditable helpers shared by the experiment caches."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def tokenizer_fingerprint(tokenizer: Any) -> dict[str, Any]:
    """Describe tokenizer behavior without tying a cache to a local path.

    Fast tokenizers expose their complete backend as deterministic JSON. Their
    most recent padding/truncation request is mutable runtime state, so it is
    excluded while the vocabulary, model, normalizer, and pre-tokenizer remain
    covered. The vocabulary fallback keeps lightweight test tokenizers
    cacheable too.
    """

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        backend_config = json.loads(backend.to_str())
        backend_config.pop("padding", None)
        backend_config.pop("truncation", None)
        serialized = json.dumps(backend_config, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    elif hasattr(tokenizer, "get_vocab"):
        vocabulary = tokenizer.get_vocab()
        serialized = json.dumps(sorted(vocabulary.items()), separators=(",", ":"), ensure_ascii=False)
    else:
        raise TypeError("Tokenizer must expose backend_tokenizer.to_str() or get_vocab() to use the corpus cache")

    try:
        vocabulary_size = len(tokenizer)
    except TypeError:
        vocabulary_size = len(tokenizer.get_vocab())
    return {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "implementation_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "special_ids": sorted(int(token_id) for token_id in tokenizer.all_special_ids),
        "vocabulary_size": vocabulary_size,
    }


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read cache file {path}: {error}") from error


def json_sha256(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_json_atomically(path: Path, value: Any) -> None:
    """Publish a complete cache file in one rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
