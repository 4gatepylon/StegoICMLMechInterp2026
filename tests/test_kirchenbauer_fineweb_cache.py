"""Tests for the minimal FineWeb cache interface.

The suite partitions cache construction across a one-document lower boundary,
a multi-part cache, exhausted input, schema-preserving metadata, default and
explicit CLI arguments, successful preview output, handled-error separation,
and missing, incomplete, obsolete, malformed, and undersized outputs. It uses
synthetic source rows and temporary artifact directories. It does not contact
FineWeb, measure shuffle quality, benchmark large caches, reproduce native
PyArrow shutdown crashes, or exercise shared multi-node filesystems.
"""

import json
from pathlib import Path

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


def test_undersized_cache_reports_default_rebuild_size(artifacts_directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del artifacts_directory
    _install_source(monkeypatch, [_source_document(0)])
    build_fineweb_cache(cache_name="undersized", documents=1)

    with pytest.raises(RuntimeError) as error:
        load_fineweb_cache("undersized", minimum_documents=321_000)

    assert "contains 1 documents but at least 321000 are required" in str(error.value)
    assert "--documents 500000" in str(error.value)


@pytest.mark.parametrize("cache_state", ["missing", "incomplete"])
def test_unavailable_cache_reports_build_command(artifacts_directory: Path, cache_state: str) -> None:
    if cache_state == "incomplete":
        (artifacts_directory / "datasets" / "fineweb" / "unavailable").mkdir(parents=True)

    with pytest.raises(FileNotFoundError) as error:
        load_fineweb_cache("unavailable")

    assert "python -m ciphers.kirchenbauer_et_al.src.cache_fineweb" in str(error.value)
    assert "--cache-name unavailable --documents 500000" in str(error.value)


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


def test_cli_defaults_are_visible() -> None:
    result = CliRunner().invoke(cache_fineweb.main, ["--help"])

    assert result.exit_code == 0
    assert "fineweb-500k" in result.output
    assert "500000" in result.output


def test_cli_verifies_and_previews_cached_metadata(artifacts_directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_source(monkeypatch, [_source_document(0), _source_document(1)])

    result = CliRunner().invoke(
        cache_fineweb.main,
        ["--cache-name", "cli-cache", "--documents", "2"],
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


def test_cli_prints_separator_before_handled_error(artifacts_directory: Path) -> None:
    existing_cache = artifacts_directory / "datasets" / "fineweb" / "existing"
    existing_cache.mkdir(parents=True)

    result = CliRunner().invoke(cache_fineweb.main, ["--cache-name", "existing", "--documents", "1"])

    assert result.exit_code != 0
    assert result.output.index("=" * 100) < result.output.index("Error: FineWeb cache already exists")


def test_inspection_notebook_is_valid_and_uses_verified_loader() -> None:
    """Cover notebook structure and syntax; model downloads and cell execution are omitted."""
    notebook_path = Path("ciphers/kirchenbauer_et_al/src/inspect_fineweb_cache.ipynb")
    notebook = json.loads(notebook_path.read_text())
    code_sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    notebook_source = "\n".join(code_sources)

    assert notebook["nbformat"] == 4
    assert "load_fineweb_cache(CACHE_NAME)" in notebook_source
    assert "token_counter.most_common(100)" in notebook_source
    assert 'document["url"]' in notebook_source
    for cell_index, code_source in enumerate(code_sources):
        compile(code_source, f"{notebook_path}:cell-{cell_index}", "exec")
