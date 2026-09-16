"""Tests for the minimal FineWeb cache interface.

The suite partitions cache construction across a one-document lower boundary,
a multi-part cache, exhausted input, schema-preserving metadata, successful CLI
execution, handled-error separation, and missing, incomplete, obsolete,
malformed, and undersized outputs. It uses
synthetic source rows and temporary artifact directories. It does not contact
FineWeb, measure shuffle quality, benchmark large caches, reproduce native
PyArrow shutdown crashes, execute the inspection notebook, or exercise shared
multi-node filesystems.

Token filtering covers unbounded, one-sided, inclusive two-sided, and zero-only
ranges, invalid bounds, filtered exhaustion, old manifests, and deterministic
splits. Remote source reads are mocked; local Parquet IO is real.

Nested Qwen filtering covers disabled, one-sided, two-sided and zero-only
bounds, GPT-2 preselection, full and partial tokenizer batches, insufficient
survivors, missing tokenizers, invalid configuration, and repeatable splits.
The Qwen tokenizer is controlled locally; real-model tokenization accuracy,
large-cache throughput and distributed training are outside this test space.

TODO(hadriano) this Codex-written test suite often looks for specific substrings in arguments, which might brittle.
"""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from ciphers.kirchenbauer_et_al.src import cache_fineweb
from ciphers.kirchenbauer_et_al.src.cache_fineweb import build_fineweb_cache, load_fineweb_cache


def _source_document(index: int) -> dict[str, object]:
    """Return one complete synthetic FineWeb row plus an ignored source field."""
    return {
        "text": f"document {index}",
        "id": f"urn:uuid:{index}",
        "dump": "CC-MAIN-2024-10",
        "url": f"https://example.com/{index}",
        "date": "2024-03-01T00:00:00Z",
        "file_path": f"s3://commoncrawl/example-{index}.warc.gz",
        "language": "en",
        "language_score": 0.99,
        "token_count": index + 1,
        "discarded": "not part of the cache schema",
    }


def _cached_document(index: int) -> dict[str, object]:
    """Return the exact nine-field row expected from the public loader."""
    source_document = _source_document(index)
    source_document.pop("discarded")
    return source_document


@pytest.fixture
def artifacts_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    return tmp_path


def _install_source(monkeypatch: pytest.MonkeyPatch, source_documents: list[dict[str, object]]) -> None:
    """Replace only the remote producer while retaining real local Parquet IO."""
    monkeypatch.setattr(cache_fineweb, "_load_remote_documents", lambda: iter(source_documents))


@pytest.mark.parametrize(
    ("documents", "part_documents", "expected_parts"),
    [(1, 10, 1), (5, 2, 3)],
)
def test_cache_contains_exact_requested_documents_and_metadata(
    artifacts_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    documents: int,
    part_documents: int,
    expected_parts: int,
) -> None:
    _install_source(monkeypatch, [_source_document(index) for index in range(documents + 1)])

    cache_directory = build_fineweb_cache(
        cache_name=f"cache-{documents}",
        documents=documents,
        part_documents=part_documents,
    )
    cached_documents = list(load_fineweb_cache(f"cache-{documents}", shuffle=False))
    manifest = json.loads((cache_directory / "manifest.json").read_text())

    assert cache_directory == artifacts_directory / "datasets" / "fineweb" / f"cache-{documents}"
    assert len(list(cache_directory.glob("part-*.parquet"))) == expected_parts
    assert manifest["format_version"] == 2
    assert manifest["documents"] == documents
    assert manifest["part_documents"] == part_documents
    assert cached_documents == [_cached_document(index) for index in range(documents)]


def test_exhausted_source_does_not_publish_cache(artifacts_directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_source(monkeypatch, [_source_document(0), _source_document(1)])

    with pytest.raises(RuntimeError, match="2 of 3 requested documents"):
        build_fineweb_cache(cache_name="exhausted", documents=3)

    assert not (artifacts_directory / "datasets" / "fineweb" / "exhausted").exists()


def test_loader_rejects_undersized_cache(artifacts_directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del artifacts_directory
    _install_source(monkeypatch, [_source_document(0)])
    build_fineweb_cache(cache_name="undersized", documents=1)

    with pytest.raises(RuntimeError) as error:
        load_fineweb_cache("undersized", minimum_documents=2)

    assert "contains 1 documents but at least 2 are required" in str(error.value)


@pytest.mark.parametrize("cache_state", ["missing", "incomplete"])
def test_unavailable_cache_reports_build_command(artifacts_directory: Path, cache_state: str) -> None:
    if cache_state == "incomplete":
        (artifacts_directory / "datasets" / "fineweb" / "unavailable").mkdir(parents=True)

    with pytest.raises(FileNotFoundError) as error:
        load_fineweb_cache("unavailable")

    assert "python -m ciphers.kirchenbauer_et_al.src.cache_fineweb" in str(error.value)
    assert "--cache-name unavailable" in str(error.value)


def test_missing_cache_reports_requirement_above_default(artifacts_directory: Path) -> None:
    del artifacts_directory

    with pytest.raises(FileNotFoundError) as error:
        load_fineweb_cache("large", minimum_documents=600_001)

    assert "--cache-name large --documents 600001" in str(error.value)


def test_loader_rejects_obsolete_manifest(artifacts_directory: Path) -> None:
    cache_directory = artifacts_directory / "datasets" / "fineweb" / "obsolete"
    cache_directory.mkdir(parents=True)
    (cache_directory / "_SUCCESS").touch()
    (cache_directory / "manifest.json").write_text('{"format_version": 1}')

    with pytest.raises(RuntimeError, match="invalid or obsolete manifest"):
        load_fineweb_cache("obsolete")


def test_loader_rejects_unreadable_part(artifacts_directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del artifacts_directory
    _install_source(monkeypatch, [_source_document(0)])
    cache_directory = build_fineweb_cache(cache_name="corrupt", documents=1)
    (cache_directory / "part-00000.parquet").write_text("not parquet")

    with pytest.raises(RuntimeError, match="unreadable part"):
        load_fineweb_cache("corrupt")


@pytest.mark.parametrize("cache_name", ["../escape", "/absolute", "nested/cache", ""])
def test_cache_name_cannot_escape_artifacts_directory(cache_name: str) -> None:
    with pytest.raises(ValueError):
        build_fineweb_cache(cache_name=cache_name, documents=1)


def test_cli_verifies_and_previews_cached_metadata(artifacts_directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_source(monkeypatch, [_source_document(-1), _source_document(0), _source_document(1), _source_document(2)])

    result = CliRunner().invoke(
        cache_fineweb.main,
        ["--cache-name", "cli-cache", "--documents", "2", "--min-gpt2-document-tokens", "1", "--max-gpt2-document-tokens", "2"],
    )

    assert result.exit_code == 0, result.output
    assert f"Cached 2 FineWeb documents at {artifacts_directory / 'datasets' / 'fineweb' / 'cli-cache'}" in result.output
    assert "Verification: PASS" in result.output
    assert "2 documents" in result.output
    assert "1/1 Parquet parts present" in result.output
    assert "Preview 1/2" in result.output
    assert "url: https://example.com/0" in result.output
    assert "text: 'document 0'" in result.output
    assert result.output.count("=" * 100) == 3
    # _source_document(index) defines token_count = index + 1, so the inclusive [1, 2] bounds retain indices 0 and 1.
    assert list(load_fineweb_cache("cli-cache", shuffle=False)) == [_cached_document(0), _cached_document(1)]


def test_cli_prints_separator_before_handled_error(artifacts_directory: Path) -> None:
    existing_cache = artifacts_directory / "datasets" / "fineweb" / "existing"
    existing_cache.mkdir(parents=True)

    result = CliRunner().invoke(cache_fineweb.main, ["--cache-name", "existing", "--documents", "1"])

    assert result.exit_code != 0
    assert result.output.index("=" * 100) < result.output.index("Error: FineWeb cache already exists")


@pytest.mark.parametrize(
    ("minimum", "maximum", "accepted_indices"),
    [(0, None, [-1, 0, 1, 2, 3]), (2, None, [1, 2, 3]), (0, 2, [-1, 0, 1]), (1, 3, [0, 1, 2]), (0, 0, [-1])],
)
# TODO(hadriano) this does not properly test cascading filters (load filtering != build filtering)
def test_document_token_filters_at_build_and_load(
    artifacts_directory: Path, monkeypatch: pytest.MonkeyPatch, minimum: int, maximum: int | None, accepted_indices: list[int]
) -> None:
    """Cover range partitions with mocked source rows and real multipart caches."""
    del artifacts_directory
    _install_source(monkeypatch, [_source_document(index) for index in range(-1, 4)])
    bounds = {"min_gpt2_document_tokens": minimum, "max_gpt2_document_tokens": maximum}
    filtered_directory = build_fineweb_cache("filtered", documents=len(accepted_indices), part_documents=2, **bounds)
    unfiltered_directory = build_fineweb_cache("unfiltered", documents=5, part_documents=2)
    expected = [_cached_document(index) for index in accepted_indices]

    assert list(load_fineweb_cache("filtered", shuffle=False)) == expected
    assert list(load_fineweb_cache("unfiltered", shuffle=False, minimum_documents=len(expected), **bounds)) == expected
    manifest = json.loads((filtered_directory / "manifest.json").read_text())
    assert (manifest["min_gpt2_document_tokens"], manifest["max_gpt2_document_tokens"]) == (minimum, maximum)
    manifest["min_document_tokens"] = manifest.pop("min_gpt2_document_tokens")
    manifest["max_document_tokens"] = manifest.pop("max_gpt2_document_tokens")
    (filtered_directory / "manifest.json").write_text(json.dumps(manifest))
    assert list(load_fineweb_cache("filtered", shuffle=False)) == expected

    # Version-2 caches created before filtering have neither bounds field.
    legacy_manifest = json.loads((unfiltered_directory / "manifest.json").read_text())
    legacy_manifest.pop("min_gpt2_document_tokens")
    legacy_manifest.pop("max_gpt2_document_tokens")
    (unfiltered_directory / "manifest.json").write_text(json.dumps(legacy_manifest))
    dataset = load_fineweb_cache("unfiltered", **bounds)
    shuffled = list(dataset)
    assert sorted(row["id"] for row in shuffled) == sorted(row["id"] for row in expected)
    assert list(dataset.take(1)) + list(dataset.skip(1)) == shuffled


def test_filtering_rejects_insufficient_accepted_documents(artifacts_directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover some and no matches; failed builds must leave no partial cache."""
    _install_source(monkeypatch, [_source_document(index) for index in range(3)])
    build_fineweb_cache("all", documents=3)
    for minimum, accepted_count in [(2, 2), (4, 0)]:
        with pytest.raises(RuntimeError, match=f"{accepted_count} of 3 requested documents"):
            build_fineweb_cache("exhausted", documents=3, part_documents=1, min_gpt2_document_tokens=minimum)
        assert sorted(path.name for path in (artifacts_directory / "datasets" / "fineweb").iterdir()) == ["all"]
        with pytest.raises(RuntimeError, match=f"{accepted_count} documents after token filtering"):
            load_fineweb_cache("all", minimum_documents=3, min_gpt2_document_tokens=minimum)


@pytest.mark.parametrize("bounds", [{"min_gpt2_document_tokens": -1}, {"max_gpt2_document_tokens": -1}, {"min_gpt2_document_tokens": 3, "max_gpt2_document_tokens": 2}])
def test_invalid_token_bounds_fail_before_io(bounds: dict[str, int], monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover negative and reversed ranges without filesystem or remote access."""
    from ciphers.kirchenbauer_et_al.src.configuration_kl_fineweb import PrefixKLTrainingConfig

    monkeypatch.delenv("STEGO_ARTIFACTS_DIR", raising=False)
    for entry_point in (build_fineweb_cache, load_fineweb_cache, PrefixKLTrainingConfig):
        with pytest.raises(ValueError):
            entry_point(**bounds)


@pytest.mark.parametrize("minimum,maximum", [(0, None), (2, None), (0, 2), (1, 3), (0, 0)])
def test_nested_qwen_filtering_preserves_rows_and_splits(artifacts_directory: Path, monkeypatch: pytest.MonkeyPatch, minimum: int, maximum: int | None) -> None:
    """Partition Qwen bounds on 34 GPT-2 survivors, spanning full/partial batches.

    GPT-2 and synthetic Qwen counts deliberately disagree. Verify GPT-2-rejected
    texts never reach the tokenizer and retained rows keep all original metadata.
    """
    del artifacts_directory
    _install_source(monkeypatch, [_source_document(index) for index in range(40)])
    build_fineweb_cache("nested", documents=40, part_documents=17)
    tokenizer = Mock(side_effect=lambda texts, **kwargs: {"input_ids": [[0] * (int(text.split()[-1]) % 5) for text in texts]})
    gpt2_indices = list(range(4, 38))
    expected = [_cached_document(index) for index in gpt2_indices if index % 5 >= minimum and (maximum is None or index % 5 <= maximum)]
    arguments = {
        "min_gpt2_document_tokens": 5,
        "max_gpt2_document_tokens": 38,
        "min_qwen_document_tokens": minimum,
        "max_qwen_document_tokens": maximum,
        "tokenizer": tokenizer,
        "minimum_documents": len(expected),
    }
    assert list(load_fineweb_cache("nested", shuffle=False, **arguments)) == expected
    dataset = load_fineweb_cache("nested", **arguments)
    shuffled = list(dataset)
    assert sorted(row["id"] for row in shuffled) == sorted(row["id"] for row in expected)
    assert list(dataset.take(2)) + list(dataset.skip(2)) == shuffled

    if minimum == 0 and maximum is None:
        tokenizer.assert_not_called()
    else:
        assert {text for call in tokenizer.call_args_list for text in call.args[0]} == {f"document {index}" for index in gpt2_indices}
        for call in tokenizer.call_args_list:
            assert 0 < len(call.args[0]) <= 32
            assert call.kwargs == {
                "add_special_tokens": False,
                "truncation": False,
                "padding": False,
                "return_attention_mask": False,
                "return_token_type_ids": False,
            }


@pytest.mark.parametrize("qwen_minimum", [4, 5])
def test_qwen_filter_rejects_insufficient_survivors_before_returning(artifacts_directory: Path, monkeypatch: pytest.MonkeyPatch, qwen_minimum: int) -> None:
    """Cover some and zero Qwen matches when GPT-2 alone has enough documents."""
    del artifacts_directory
    _install_source(monkeypatch, [_source_document(index) for index in range(6)])
    build_fineweb_cache("nested-small", documents=6, part_documents=2)
    tokenizer = Mock(side_effect=lambda texts, **kwargs: {"input_ids": [[0] * (int(text.split()[-1]) % 5) for text in texts]})
    with pytest.raises(RuntimeError, match="after GPT-2 and Qwen token filtering"):
        load_fineweb_cache("nested-small", min_gpt2_document_tokens=2, min_qwen_document_tokens=qwen_minimum, minimum_documents=2, tokenizer=tokenizer)


@pytest.mark.parametrize("bounds", [{"min_qwen_document_tokens": -1}, {"max_qwen_document_tokens": -1}, {"min_qwen_document_tokens": 3, "max_qwen_document_tokens": 2}])
def test_invalid_qwen_bounds_fail_before_io(bounds: dict[str, int], monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject negative and reversed Qwen bounds in both loader and training config."""
    from ciphers.kirchenbauer_et_al.src.configuration_kl_fineweb import PrefixKLTrainingConfig

    monkeypatch.delenv("STEGO_ARTIFACTS_DIR", raising=False)
    for entry_point in (load_fineweb_cache, PrefixKLTrainingConfig):
        with pytest.raises(ValueError):
            entry_point(**bounds)


@pytest.mark.parametrize("bounds", [{"min_qwen_document_tokens": 1}, {"max_qwen_document_tokens": 0}])
def test_enabled_qwen_filter_requires_tokenizer(bounds: dict[str, int], monkeypatch: pytest.MonkeyPatch) -> None:
    """Require the companion tokenizer for both lower-only and zero-only filters."""
    monkeypatch.delenv("STEGO_ARTIFACTS_DIR", raising=False)
    with pytest.raises(ValueError, match="tokenizer is required"):
        load_fineweb_cache(**bounds)
