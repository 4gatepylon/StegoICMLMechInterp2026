"""Build and load a bounded, verified local FineWeb document cache.

The module deliberately exposes only :func:`build_fineweb_cache` and
:func:`load_fineweb_cache` as its programmatic interface. With the default
arguments, the builder reads the pinned ``sample-10BT`` stream and writes
500,000 documents below
``$STEGO_ARTIFACTS_DIR/datasets/fineweb/fineweb-500k``. The command-line
interface performs the same operation when invoked without options::

    python -m ciphers.kirchenbauer_et_al.src.cache_fineweb

More formally, let ``S = (d_0, d_1, ...)`` be the sequential remote stream and
let ``F = {text, id, dump, url, date, file_path, language, language_score,
token_count}``. For a requested document count ``N``, cached row ``c_i`` is the
projection of the i-th document satisfying the inclusive token-count bounds
onto ``F``, and ``C_N = (c_0, ..., c_(N-1))``. Defaults accept every row. Field
definitions come from the FineWeb dataset card:
https://huggingface.co/datasets/HuggingFaceFW/fineweb#data-fields.

For a Parquet part capacity ``B``, construction writes ``P = ceil(N / B)``
parts; part ``j`` contains indices ``jB`` through ``min((j + 1)B, N) - 1``. A
manifest records ``N``, ``P``, ``B``, the dataset revision, and the on-disk
format. Parts and metadata are assembled in a temporary sibling directory,
``_SUCCESS`` is written last, and the directory is atomically renamed to its
final artifacts path. Loaders always verify the marker, manifest, minimum
document count, contiguous parts, Parquet schema, and row count before returning
any data.

By default, loading applies ``IterableDataset.shuffle(seed=42,
buffer_size=10_000)``. This first shuffles the Parquet shard order, fills a
rolling buffer with 10,000 examples, repeatedly emits a randomly selected
buffer member, and replaces it with the next source example. It is therefore
not a shuffle of fixed, deterministic 10,000-document blocks, but it is still
only an approximate global shuffle: documents cannot mix as freely as they
would under a uniform permutation of all ``N`` rows unless the buffer holds the
whole dataset. See the Hugging Face buffer-shuffle documentation:
https://huggingface.co/docs/datasets/stream#shuffle. The seed makes repeated
iterations deterministic unless a consumer changes the epoch with
``set_epoch``. The trainer takes the first ``V`` examples from this ordering for
validation and skips the same ``V`` examples for training.

The Hugging Face/PyArrow streaming stack can rarely abort during interpreter
finalization after an early, bounded read of the much larger remote dataset.
This occurs after Python code has returned and can produce a nonzero exit even
when the atomic cache publication succeeded. The CLI prints an explicit
verification result and a 100-character separator before shutdown diagnostics;
``Verification: PASS`` together with ``_SUCCESS`` identifies a usable cache.
"""

import os
import shutil
import sys
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal

import click
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, IterableDataset, load_dataset
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, StringConstraints, TypeAdapter, ValidationError

from ciphers.kirchenbauer_et_al.src.configuration_kl_fineweb import DocumentTokenFilter

FINEWEB_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
CACHE_FORMAT_VERSION = 2
CACHE_BUILD_MODULE = "ciphers.kirchenbauer_et_al.src.cache_fineweb"
_DEFAULT_CACHE_NAME = "fineweb-500k"
_DEFAULT_DOCUMENTS = 500_000
_DEFAULT_PART_DOCUMENTS = 10_000
_SHUFFLE_SEED = 42
_SHUFFLE_BUFFER_SIZE = 10_000
_PREVIEW_DOCUMENTS = 3
_PREVIEW_CHARACTERS = 500
_OUTPUT_SEPARATOR = "=" * 100
_CacheName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
_EXPECTED_ARROW_SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("id", pa.string()),
        ("dump", pa.string()),
        ("url", pa.string()),
        ("date", pa.string()),
        ("file_path", pa.string()),
        ("language", pa.string()),
        ("language_score", pa.float64()),
        ("token_count", pa.int64()),
    ]
)

__all__ = ["build_fineweb_cache", "load_fineweb_cache"]


class _CacheConfig(DocumentTokenFilter):
    """Validated inputs for creating one bounded FineWeb cache.

    Attributes:
        cache_name: Directory name below
            ``$STEGO_ARTIFACTS_DIR/datasets/fineweb``. Path separators are
            rejected so callers cannot place generated data outside the
            artifacts directory.
        documents: Exact number of accepted source documents to store.
        part_documents: Maximum documents per Parquet part. This bounds memory
            during construction; it does not affect the examples read later.
    """

    model_config = ConfigDict(extra="forbid")

    cache_name: _CacheName
    documents: int = Field(gt=0)
    part_documents: int = Field(default=_DEFAULT_PART_DOCUMENTS, gt=0)


class _CacheManifest(DocumentTokenFilter):
    """Schema persisted beside Parquet parts and validated by the trainer.

    The source fields pin the remote dataset identity, while ``documents`` and
    ``parts`` let the loader reject undersized or partially copied caches.
    ``format_version`` prevents a future loader from silently accepting an
    incompatible on-disk layout.
    Inherited token bounds record construction filtering; older version-2
    manifests without them use the unfiltered defaults.
    """

    model_config = ConfigDict(extra="forbid")

    format_version: Literal[CACHE_FORMAT_VERSION] = CACHE_FORMAT_VERSION
    cache_name: _CacheName
    documents: int = Field(gt=0)
    parts: int = Field(gt=0)
    part_documents: int = Field(gt=0)
    source_dataset: Literal["HuggingFaceFW/fineweb"] = "HuggingFaceFW/fineweb"
    source_config: Literal["sample-10BT"] = "sample-10BT"
    source_revision: Literal[FINEWEB_REVISION] = FINEWEB_REVISION


class _CachedDocument(BaseModel):
    """Validate the complete FineWeb row persisted in every cache part.

    ``text`` is extracted page content; ``id``, ``dump``, ``url``, ``date``,
    and ``file_path`` identify the Common Crawl source; ``language`` and
    ``language_score`` are FineWeb's language annotation; and ``token_count``
    is the length under the GPT-2 tokenizer, not the Qwen training tokenizer.
    The public builder and loader document this same nine-field boundary.
    """

    model_config = ConfigDict(extra="ignore")

    text: str
    id: str
    dump: str
    url: str
    date: str
    file_path: str
    language: str
    language_score: float = Field(ge=0.0, le=1.0)
    token_count: int = Field(ge=0)


def _cache_directory(cache_name: str) -> Path:
    """Return the artifacts-relative directory for a validated cache name.

    Args:
        cache_name: A single portable directory name, without path separators.

    Returns:
        ``$STEGO_ARTIFACTS_DIR/datasets/fineweb/<cache_name>`` as a ``Path``.

    Raises:
        KeyError: If ``STEGO_ARTIFACTS_DIR`` is not set.
        ValidationError: If ``cache_name`` could escape the cache root.
    """
    validated_name = TypeAdapter(_CacheName).validate_python(cache_name)
    return Path(os.environ["STEGO_ARTIFACTS_DIR"]) / "datasets" / "fineweb" / validated_name


def _cache_build_command(cache_name: str, documents: int = _DEFAULT_DOCUMENTS) -> str:
    """Return the single-process command used to create a missing cache.

    Args:
        cache_name: Validated cache name passed through to the Click command.
        documents: Exact positive document count to pass to the builder.

    Returns:
        A shell command that invokes this module from the repository root.
    """
    validated_name = TypeAdapter(_CacheName).validate_python(cache_name)
    validated_documents = TypeAdapter(PositiveInt).validate_python(documents)
    return f"python -m {CACHE_BUILD_MODULE} --cache-name {validated_name} --documents {validated_documents}"


def _load_remote_documents() -> Iterable[Mapping[str, object]]:
    """Return the pinned, sequential FineWeb stream used only during builds.

    Returns:
        An iterable whose rows must satisfy ``_CachedDocument``: ``text``,
        ``id``, ``dump``, ``url``, ``date``, ``file_path``, ``language``,
        ``language_score``, and ``token_count``. ``build_fineweb_cache`` keeps
        exactly those fields and rejects rows that violate the schema.
    """
    return load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
        revision=FINEWEB_REVISION,
    )


def _write_parquet_part(
    temporary_directory: Path,
    part_index: int,
    cached_documents: Sequence[Mapping[str, object]],
) -> None:
    """Write one nonempty, schema-validated batch to a numbered Parquet part.

    Args:
        temporary_directory: Build directory that is not published until every
            part and completion marker exists.
        part_index: Zero-based index used in ``part-00000.parquet`` names.
        cached_documents: Nonempty sequence of dictionaries matching the nine
            fields documented by ``_CachedDocument``. The caller clears its
            mutable buffer only after this function returns successfully.

    Returns:
        ``None``. The sole output is ``part-{part_index:05d}.parquet`` inside
        ``temporary_directory``.
    """
    if not cached_documents:
        raise ValueError("cannot write an empty FineWeb cache part")
    part_path = temporary_directory / f"part-{part_index:05d}.parquet"
    Dataset.from_list(list(cached_documents)).to_parquet(str(part_path))


def _validate_cache(
    cache_name: str,
    *,
    minimum_documents: int = 1,
    exact_documents: int | None = None,
) -> tuple[Path, _CacheManifest, list[Path]]:
    """Validate every on-disk invariant before a cache is reported or loaded.

    Args:
        cache_name: Directory name below the artifacts cache root.
        minimum_documents: Smallest manifest count accepted by a caller.
        exact_documents: If provided, require this exact manifest count. The
            builder CLI uses this stronger check before reporting success.

    Returns:
        A three-tuple containing the cache directory, parsed ``_CacheManifest``,
        and sorted Parquet part paths. Callers use the paths to load data and
        the manifest to report the verified document and part counts.

    Raises:
        FileNotFoundError: If the directory or ``_SUCCESS`` marker is absent.
        RuntimeError: If the manifest is incompatible, counts do not satisfy
            the caller, or the numbered Parquet parts are incomplete.
        ValidationError: If an input name or count is invalid.
    """
    minimum_documents = TypeAdapter(PositiveInt).validate_python(minimum_documents)
    if exact_documents is not None:
        exact_documents = TypeAdapter(PositiveInt).validate_python(exact_documents)
    cache_directory = _cache_directory(cache_name)
    manifest_path = cache_directory / "manifest.json"
    success_path = cache_directory / "_SUCCESS"
    suggested_documents = max(_DEFAULT_DOCUMENTS, minimum_documents)
    build_command = _cache_build_command(cache_name, suggested_documents)
    if not manifest_path.is_file() or not success_path.is_file():
        raise FileNotFoundError(f"FineWeb cache '{cache_name}' is missing or incomplete. Create it before training:\n{build_command}")
    try:
        manifest = _CacheManifest.model_validate_json(manifest_path.read_text())
    except (OSError, ValidationError) as error:
        raise RuntimeError(f"FineWeb cache '{cache_name}' has an invalid or obsolete manifest. Move it aside and rebuild it with:\n{build_command}") from error
    if manifest.cache_name != cache_name or manifest.documents < minimum_documents:
        raise RuntimeError(
            f"FineWeb cache '{cache_name}' contains {manifest.documents} documents but at least {minimum_documents} are required. "
            f"Move it aside and rebuild it with:\n{build_command}"
        )
    if exact_documents is not None and manifest.documents != exact_documents:
        raise RuntimeError(f"FineWeb cache '{cache_name}' reports {manifest.documents} documents; expected exactly {exact_documents}")
    expected_parts = (manifest.documents + manifest.part_documents - 1) // manifest.part_documents
    if manifest.parts != expected_parts:
        raise RuntimeError(
            f"FineWeb cache '{cache_name}' manifest reports {manifest.parts} parts but its counts require {expected_parts}. Move it aside and rebuild it with:\n{build_command}"
        )
    part_paths = sorted(cache_directory.glob("part-*.parquet"))
    if len(part_paths) != manifest.parts:
        raise RuntimeError(f"FineWeb cache '{cache_name}' has {len(part_paths)} of {manifest.parts} Parquet parts. Move it aside and rebuild it with:\n{build_command}")
    cached_document_count = 0
    for part_index, part_path in enumerate(part_paths):
        expected_part_path = cache_directory / f"part-{part_index:05d}.parquet"
        if part_path != expected_part_path:
            raise RuntimeError(f"FineWeb cache '{cache_name}' has a noncontiguous part sequence at {part_path.name}. Move it aside and rebuild it with:\n{build_command}")
        try:
            parquet_file = pq.ParquetFile(part_path)
        except (OSError, pa.ArrowException) as error:
            raise RuntimeError(f"FineWeb cache '{cache_name}' contains an unreadable part {part_path.name}. Move it aside and rebuild it with:\n{build_command}") from error
        if not parquet_file.schema_arrow.remove_metadata().equals(_EXPECTED_ARROW_SCHEMA):
            raise RuntimeError(f"FineWeb cache '{cache_name}' part {part_path.name} has an incompatible document schema. Move it aside and rebuild it with:\n{build_command}")
        expected_part_documents = min(manifest.part_documents, manifest.documents - cached_document_count)
        if parquet_file.metadata.num_rows != expected_part_documents:
            raise RuntimeError(
                f"FineWeb cache '{cache_name}' part {part_path.name} contains {parquet_file.metadata.num_rows} rows; "
                f"expected {expected_part_documents}. Move it aside and rebuild it with:\n{build_command}"
            )
        cached_document_count += parquet_file.metadata.num_rows
    if cached_document_count != manifest.documents:
        raise RuntimeError(
            f"FineWeb cache '{cache_name}' parts contain {cached_document_count} rows but the manifest reports "
            f"{manifest.documents}. Move it aside and rebuild it with:\n{build_command}"
        )
    return cache_directory, manifest, part_paths


def build_fineweb_cache(
    cache_name: str = _DEFAULT_CACHE_NAME,
    documents: int = _DEFAULT_DOCUMENTS,
    *,
    part_documents: int = _DEFAULT_PART_DOCUMENTS,
    min_document_tokens: int = 0,
    max_document_tokens: int | None = None,
) -> Path:
    """Materialize a bounded prefix of accepted FineWeb rows with metadata.

    Args:
        cache_name: Directory name below
            ``$STEGO_ARTIFACTS_DIR/datasets/fineweb``. It defaults to
            ``fineweb-500k`` and cannot contain path separators.
        documents: Exact number of accepted sequential ``sample-10BT`` rows to
            retain; defaults to 500,000. Rejected rows do not count.
        part_documents: Maximum rows per Parquet part; defaults to 10,000 and
            therefore produces 50 parts for the default build.
        min_document_tokens: Inclusive minimum stored GPT-2 token count, before
            training tokenization or truncation. Zero accepts empty documents.
        max_document_tokens: Inclusive maximum stored GPT-2 token count;
            ``None`` means no upper bound. Must be at least the minimum.

    Returns:
        The completed cache directory. It contains ``part-*.parquet`` files,
        ``manifest.json``, and a ``_SUCCESS`` marker. The trainer requires all
        three forms of output before it will read the cache. Each dataset row
        contains ``text: str``, ``id: str``, ``dump: str``, ``url: str``,
        ``date: str``, ``file_path: str``, ``language: str``,
        ``language_score: float``, and ``token_count: int``.

    Raises:
        FileExistsError: If the destination already exists.
        ValidationError: If configuration or a source row is invalid.
        RuntimeError: If the source ends before the requested document count.

    The remote source is intentionally not shuffled: taking sequential source
    rows avoids opening many remote shards while downloading. The loader
    shuffles the completed local cache instead.
    """
    config = _CacheConfig(
        cache_name=cache_name,
        documents=documents,
        part_documents=part_documents,
        min_document_tokens=min_document_tokens,
        max_document_tokens=max_document_tokens,
    )
    cache_directory = _cache_directory(config.cache_name)
    if cache_directory.exists():
        raise FileExistsError(f"FineWeb cache already exists: {cache_directory}")
    cache_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(tempfile.mkdtemp(prefix=f".{config.cache_name}-", dir=cache_directory.parent))
    document_count = 0
    part_count = 0
    document_buffer: list[dict[str, object]] = []

    try:
        source_iterator: Iterator[Mapping[str, object]] = iter(_load_remote_documents())
        try:
            for source_document in source_iterator:
                cached_document = _CachedDocument.model_validate(source_document)
                if not config.accepts(cached_document.token_count):
                    continue
                document_buffer.append(cached_document.model_dump())
                document_count += 1
                if len(document_buffer) == config.part_documents:
                    _write_parquet_part(temporary_directory, part_count, document_buffer)
                    document_buffer.clear()
                    part_count += 1
                if document_count == config.documents:
                    break
        finally:
            close_source = getattr(source_iterator, "close", None)
            if callable(close_source):
                close_source()
        if document_count != config.documents:
            raise RuntimeError(f"FineWeb source ended after {document_count} of {config.documents} requested documents")
        if document_buffer:
            _write_parquet_part(temporary_directory, part_count, document_buffer)
            document_buffer.clear()
            part_count += 1
        manifest = _CacheManifest(
            cache_name=config.cache_name,
            documents=document_count,
            parts=part_count,
            part_documents=config.part_documents,
            min_document_tokens=config.min_document_tokens,
            max_document_tokens=config.max_document_tokens,
        )
        (temporary_directory / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        (temporary_directory / "_SUCCESS").touch()
        temporary_directory.rename(cache_directory)
    except BaseException:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    _validate_cache(config.cache_name, exact_documents=config.documents)
    return cache_directory


def load_fineweb_cache(
    cache_name: str = _DEFAULT_CACHE_NAME,
    *,
    minimum_documents: int = 1,
    shuffle: bool = True,
    min_document_tokens: int = 0,
    max_document_tokens: int | None = None,
) -> IterableDataset:
    """Load a completed local cache for the KL trainer.

    Args:
        cache_name: Validated cache directory name below the artifacts root.
        minimum_documents: Smallest acceptable count after filtering. The
            trainer uses this to reserve validation plus training data.
        shuffle: Whether to apply the deterministic local shuffle. Production
            callers retain the default; tests may disable it to inspect order.
        min_document_tokens: Inclusive minimum stored GPT-2 token count. With
            either bound enabled, scan only the Parquet token-count column to
            check the accepted count before returning the streaming dataset.
        max_document_tokens: Inclusive maximum stored GPT-2 token count;
            ``None`` disables the upper bound. Loading filters are independent
            of construction bounds and cannot recover documents omitted then.

    Returns:
        A streaming local ``IterableDataset``. Every row contains the nine keys
        documented by ``build_fineweb_cache``. The trainer consumes ``text``;
        inspection tools also use the document provenance and annotations.
        Filtering precedes shuffle and the caller's train/validation split.

    Raises:
        FileNotFoundError: If the cache is absent or lacks its completion marker.
        RuntimeError: If the manifest or Parquet parts are invalid or too small.
    """
    token_filter = DocumentTokenFilter(min_document_tokens=min_document_tokens, max_document_tokens=max_document_tokens)
    _, _, part_paths = _validate_cache(cache_name, minimum_documents=minimum_documents)
    filtering = token_filter.min_document_tokens > 0 or token_filter.max_document_tokens is not None
    if filtering:
        accepted_documents = sum(
            token_filter.accepts(token_count)
            for part_path in part_paths
            for batch in pq.ParquetFile(part_path).iter_batches(columns=["token_count"])
            for token_count in batch.column(0).to_pylist()
        )
        if accepted_documents < minimum_documents:
            raise RuntimeError(
                f"FineWeb cache '{cache_name}' contains {accepted_documents} documents after token filtering "
                f"but at least {minimum_documents} are required. Relax the bounds or build a larger cache."
            )
    dataset = load_dataset(
        "parquet",
        data_files={"train": [str(part_path) for part_path in part_paths]},
        split="train",
        streaming=True,
    )
    if filtering:
        dataset = dataset.filter(token_filter.accepts, input_columns=["token_count"])
    return dataset.shuffle(seed=_SHUFFLE_SEED, buffer_size=_SHUFFLE_BUFFER_SIZE) if shuffle else dataset


def _preview_cache(cache_name: str, documents: int = _PREVIEW_DOCUMENTS) -> list[dict[str, object]]:
    """Load and schema-check a small unshuffled preview from a verified cache.

    Args:
        cache_name: Cache passed through the public, always-verifying loader.
        documents: Maximum number of leading cached rows to return.

    Returns:
        Up to ``documents`` dictionaries. Every dictionary has exactly the nine
        keys documented by ``build_fineweb_cache`` and is suitable for the CLI
        metadata and text preview.
    """
    documents = TypeAdapter(PositiveInt).validate_python(documents)
    cached_dataset = load_fineweb_cache(cache_name, shuffle=False)
    return [_CachedDocument.model_validate(document).model_dump() for document in cached_dataset.take(documents)]


def _format_preview_text(text: str) -> str:
    """Return a one-line, visibly truncated representation for CLI previews."""
    preview_text = text[:_PREVIEW_CHARACTERS]
    suffix = " ... [truncated]" if len(text) > _PREVIEW_CHARACTERS else ""
    return repr(preview_text) + suffix


@click.command()
@click.option(
    "--cache-name",
    default=_DEFAULT_CACHE_NAME,
    show_default=True,
    help="Name below $STEGO_ARTIFACTS_DIR/datasets/fineweb.",
)
@click.option(
    "--documents",
    default=_DEFAULT_DOCUMENTS,
    show_default=True,
    type=click.IntRange(min=1),
    help="Exact number of documents to store.",
)
@click.option("--min-document-tokens", default=0, show_default=True, type=click.IntRange(min=0), help="Inclusive minimum stored GPT-2 token count.")
@click.option("--max-document-tokens", default=None, type=click.IntRange(min=0), help="Inclusive maximum stored GPT-2 token count; omitted means unlimited.")
def main(cache_name: str, documents: int, min_document_tokens: int, max_document_tokens: int | None) -> None:
    """Build, verify, and preview a bounded FineWeb cache in local artifacts."""
    try:
        cache_directory = build_fineweb_cache(cache_name, documents, min_document_tokens=min_document_tokens, max_document_tokens=max_document_tokens)
        _, manifest, part_paths = _validate_cache(cache_name, exact_documents=documents)
        preview_documents = _preview_cache(cache_name)
    except (FileExistsError, KeyError, OSError, RuntimeError, ValueError, ValidationError) as error:
        click.echo(_OUTPUT_SEPARATOR, err=True)
        raise click.ClickException(str(error)) from error
    click.echo(f"Cached {documents} FineWeb documents at {cache_directory}")
    click.echo(
        f"Verification: PASS — directory exists, _SUCCESS present, manifest reports {manifest.documents} documents, {len(part_paths)}/{manifest.parts} Parquet parts present"
    )
    click.echo(_OUTPUT_SEPARATOR)
    for preview_index, cached_document in enumerate(preview_documents, start=1):
        click.echo(f"Preview {preview_index}/{len(preview_documents)}")
        for metadata_key in ("id", "dump", "url", "date", "file_path", "language", "language_score", "token_count"):
            click.echo(f"{metadata_key}: {cached_document[metadata_key]}")
        click.echo(f"text: {_format_preview_text(str(cached_document['text']))}")
        click.echo(_OUTPUT_SEPARATOR)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
