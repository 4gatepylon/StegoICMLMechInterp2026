"""Build and load bounded local FineWeb document caches."""

import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Literal

import click
from datasets import Dataset, IterableDataset, load_dataset
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, StringConstraints, TypeAdapter, ValidationError

FINEWEB_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
CACHE_FORMAT_VERSION = 1
CACHE_BUILD_MODULE = "ciphers.kirchenbauer_et_al.src.cache_fineweb"
CacheName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]


class FineWebCacheConfig(BaseModel):
    """Validated inputs for creating one bounded FineWeb cache.

    Attributes:
        cache_name: Directory name below
            ``$STEGO_ARTIFACTS_DIR/datasets/fineweb``. Path separators are
            rejected so callers cannot place generated data outside the
            artifacts directory.
        documents: Exact number of source documents to store.
        part_documents: Maximum documents per Parquet part. This bounds memory
            during construction; it does not affect the examples read later.
    """

    model_config = ConfigDict(extra="forbid")

    cache_name: CacheName
    documents: int = Field(gt=0)
    part_documents: int = Field(default=10_000, gt=0)


class FineWebCacheManifest(BaseModel):
    """Schema persisted beside Parquet parts and validated by the trainer.

    The source fields pin the remote dataset identity, while ``documents`` and
    ``parts`` let the loader reject undersized or partially copied caches.
    ``format_version`` prevents a future loader from silently accepting an
    incompatible on-disk layout.
    """

    model_config = ConfigDict(extra="forbid")

    format_version: Literal[CACHE_FORMAT_VERSION] = CACHE_FORMAT_VERSION
    cache_name: CacheName
    documents: int = Field(gt=0)
    parts: int = Field(gt=0)
    source_dataset: Literal["HuggingFaceFW/fineweb"] = "HuggingFaceFW/fineweb"
    source_config: Literal["sample-10BT"] = "sample-10BT"
    source_revision: Literal[FINEWEB_REVISION] = FINEWEB_REVISION


def fineweb_cache_directory(cache_name: str) -> Path:
    """Return the artifacts-relative directory for a validated cache name.

    Args:
        cache_name: A single portable directory name, without path separators.

    Returns:
        ``$STEGO_ARTIFACTS_DIR/datasets/fineweb/<cache_name>`` as a ``Path``.

    Raises:
        KeyError: If ``STEGO_ARTIFACTS_DIR`` is not set.
        ValidationError: If ``cache_name`` could escape the cache root.
    """
    validated_name = TypeAdapter(CacheName).validate_python(cache_name)
    return Path(os.environ["STEGO_ARTIFACTS_DIR"]) / "datasets" / "fineweb" / validated_name


def cache_build_command(cache_name: str, documents: int | None = None) -> str:
    """Return the single-process command used to create a missing cache.

    Args:
        cache_name: Validated cache name passed through to the Click command.
        documents: Requested count, or ``None`` to emit ``DOCUMENT_COUNT`` as a
            visible placeholder for callers that do not know the required size.

    Returns:
        A shell command that invokes this module from the repository root.
    """
    validated_name = TypeAdapter(CacheName).validate_python(cache_name)
    document_argument = str(documents) if documents is not None else "DOCUMENT_COUNT"
    return f"python -m {CACHE_BUILD_MODULE} --cache-name {validated_name} --documents {document_argument}"


def build_fineweb_cache(
    config: FineWebCacheConfig,
    source_documents: Iterable[Mapping[str, object]] | None = None,
) -> Path:
    """Materialize exactly ``config.documents`` raw texts into local Parquet.

    Args:
        config: Validated cache name, document count, and Parquet part size.
        source_documents: Optional iterable used instead of the remote dataset.
            Every mapping must contain a string ``text`` value; other keys are
            discarded. Tests use this injection point to avoid network access.

    Returns:
        The completed cache directory. It contains ``part-*.parquet`` files,
        ``manifest.json``, and a ``_SUCCESS`` marker. The trainer requires all
        three forms of output before it will read the cache.

    Raises:
        FileExistsError: If the destination already exists.
        ValueError: If a source row lacks a string ``text`` value.
        RuntimeError: If the source ends before the requested document count.

    The remote source is intentionally not shuffled: taking sequential source
    rows avoids opening many remote shards while downloading. The loader
    shuffles the completed local cache instead.
    """
    cache_directory = fineweb_cache_directory(config.cache_name)
    if cache_directory.exists():
        raise FileExistsError(f"FineWeb cache already exists: {cache_directory}")
    cache_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(tempfile.mkdtemp(prefix=f".{config.cache_name}-", dir=cache_directory.parent))
    if source_documents is None:
        source_documents = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True,
            revision=FINEWEB_REVISION,
        )
    document_count = 0
    part_count = 0
    text_buffer: list[str] = []

    def write_part() -> None:
        nonlocal part_count
        if not text_buffer:
            return
        part_path = temporary_directory / f"part-{part_count:05d}.parquet"
        Dataset.from_dict({"text": list(text_buffer)}).to_parquet(str(part_path))
        text_buffer.clear()
        part_count += 1

    try:
        for source_document in source_documents:
            text = source_document.get("text")
            if not isinstance(text, str):
                raise ValueError("each FineWeb source document must contain a string 'text' value")
            text_buffer.append(text)
            document_count += 1
            if len(text_buffer) == config.part_documents:
                write_part()
            if document_count == config.documents:
                break
        if document_count != config.documents:
            raise RuntimeError(f"FineWeb source ended after {document_count} of {config.documents} requested documents")
        write_part()
        manifest = FineWebCacheManifest(
            cache_name=config.cache_name,
            documents=document_count,
            parts=part_count,
        )
        (temporary_directory / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        (temporary_directory / "_SUCCESS").touch()
        temporary_directory.rename(cache_directory)
    except BaseException:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    return cache_directory


def load_fineweb_cache(
    cache_name: str,
    *,
    minimum_documents: int = 1,
    shuffle: bool = True,
) -> IterableDataset:
    """Load a completed local cache for the KL trainer.

    Args:
        cache_name: Validated cache directory name below the artifacts root.
        minimum_documents: Smallest acceptable manifest count. The trainer uses
            this to reserve its fixed validation prefix plus training data.
        shuffle: Whether to apply the deterministic local shuffle. Production
            callers retain the default; tests may disable it to inspect order.

    Returns:
        A streaming local ``IterableDataset`` whose rows have exactly one key,
        ``text: str``. The trainer splits this iterable into fixed validation
        rows and the remaining training rows.

    Raises:
        FileNotFoundError: If the cache is absent or lacks its completion marker.
        RuntimeError: If the manifest or Parquet parts are invalid or too small.
    """
    minimum_documents = TypeAdapter(PositiveInt).validate_python(minimum_documents)
    cache_directory = fineweb_cache_directory(cache_name)
    manifest_path = cache_directory / "manifest.json"
    success_path = cache_directory / "_SUCCESS"
    suggested_documents = max(100_000, minimum_documents)
    build_command = cache_build_command(cache_name, suggested_documents)
    if not manifest_path.is_file() or not success_path.is_file():
        raise FileNotFoundError(f"FineWeb cache '{cache_name}' is missing or incomplete. Create it before training:\n{build_command}")
    try:
        manifest = FineWebCacheManifest.model_validate_json(manifest_path.read_text())
    except (OSError, ValidationError) as error:
        raise RuntimeError(f"FineWeb cache '{cache_name}' has an invalid manifest. Rebuild it with:\n{build_command}") from error
    if manifest.cache_name != cache_name or manifest.documents < minimum_documents:
        raise RuntimeError(f"FineWeb cache '{cache_name}' contains {manifest.documents} documents but at least {minimum_documents} are required. Rebuild it with:\n{build_command}")
    part_paths = sorted(cache_directory.glob("part-*.parquet"))
    if len(part_paths) != manifest.parts:
        raise RuntimeError(f"FineWeb cache '{cache_name}' has missing Parquet parts. Rebuild it with:\n{build_command}")
    dataset = load_dataset(
        "parquet",
        data_files={"train": [str(part_path) for part_path in part_paths]},
        split="train",
        streaming=True,
    )
    return dataset.shuffle(seed=42, buffer_size=10_000) if shuffle else dataset


@click.command()
@click.option("--cache-name", required=True, help="Name below $STEGO_ARTIFACTS_DIR/datasets/fineweb.")
@click.option("--documents", required=True, type=click.IntRange(min=1), help="Exact number of documents to store.")
def main(cache_name: str, documents: int) -> None:
    """Download a bounded number of FineWeb documents into local artifacts."""
    try:
        config = FineWebCacheConfig(cache_name=cache_name, documents=documents)
        cache_directory = build_fineweb_cache(config)
    except (FileExistsError, RuntimeError, ValueError, ValidationError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Cached {documents} FineWeb documents at {cache_directory}")


if __name__ == "__main__":
    main()
