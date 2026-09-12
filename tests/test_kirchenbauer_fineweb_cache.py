"""Tests for bounded FineWeb caching.

The suite partitions cache construction across a one-document lower boundary,
a multi-part cache, exhausted input, missing/incomplete/undersized outputs, and
invalid names. It uses only synthetic documents and temporary artifact
directories; it does not contact FineWeb, benchmark large caches, or exercise
shared multi-node filesystems.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ciphers.kirchenbauer_et_al.binary_classification_mvp.cache_fineweb import (
    FineWebCacheConfig,
    FineWebCacheManifest,
    build_fineweb_cache,
    fineweb_cache_directory,
    load_fineweb_cache,
)


@pytest.fixture
def artifacts_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize(
    ("documents", "part_documents", "expected_parts"),
    [(1, 10, 1), (5, 2, 3)],
)
def test_cache_contains_exact_requested_documents(
    artifacts_directory: Path,
    documents: int,
    part_documents: int,
    expected_parts: int,
) -> None:
    config = FineWebCacheConfig(
        cache_name=f"cache-{documents}",
        documents=documents,
        part_documents=part_documents,
    )
    source_documents = ({"text": f"document {index}", "discarded": index} for index in range(documents + 1))

    cache_directory = build_fineweb_cache(config, source_documents)
    cached_documents = list(load_fineweb_cache(config.cache_name, shuffle=False))
    manifest = FineWebCacheManifest.model_validate_json((cache_directory / "manifest.json").read_text())

    assert cache_directory.is_relative_to(artifacts_directory)
    assert len(list(cache_directory.glob("part-*.parquet"))) == expected_parts
    assert manifest.documents == documents
    assert cached_documents == [{"text": f"document {index}"} for index in range(documents)]


def test_exhausted_source_does_not_publish_cache(artifacts_directory: Path) -> None:
    config = FineWebCacheConfig(cache_name="exhausted", documents=3)

    with pytest.raises(RuntimeError, match="2 of 3 requested documents"):
        build_fineweb_cache(config, [{"text": "first"}, {"text": "second"}])

    assert not fineweb_cache_directory(config.cache_name).exists()


def test_undersized_cache_reports_required_document_count(artifacts_directory: Path) -> None:
    config = FineWebCacheConfig(cache_name="undersized", documents=1)
    build_fineweb_cache(config, [{"text": "only document"}])

    with pytest.raises(RuntimeError) as error:
        load_fineweb_cache(config.cache_name, minimum_documents=321_000)

    assert "contains 1 documents but at least 321000 are required" in str(error.value)
    assert "--documents 321000" in str(error.value)


@pytest.mark.parametrize("cache_state", ["missing", "incomplete"])
def test_unavailable_cache_reports_build_command(
    artifacts_directory: Path,
    cache_state: str,
) -> None:
    del artifacts_directory
    if cache_state == "incomplete":
        fineweb_cache_directory("unavailable").mkdir(parents=True)

    with pytest.raises(FileNotFoundError) as error:
        load_fineweb_cache("unavailable")

    assert "python -m ciphers.kirchenbauer_et_al.binary_classification_mvp.cache_fineweb" in str(error.value)
    assert "--cache-name unavailable --documents 100000" in str(error.value)


@pytest.mark.parametrize("cache_name", ["../escape", "/absolute", "nested/cache", ""])
def test_cache_name_cannot_escape_artifacts_directory(cache_name: str) -> None:
    with pytest.raises(ValidationError):
        FineWebCacheConfig(cache_name=cache_name, documents=1)
