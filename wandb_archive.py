#!/usr/bin/env python
"""Mark W&B runs for this repository and archive terminal marked runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

import click
import wandb
from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parent
ARCHIVE_ROOT = REPO_ROOT / "wandb_runs"
ARCHIVE_TAG = "stego-icml-2026-git-archive"
MAX_ARCHIVE_BYTES = 95 * 1024 * 1024
DOWNLOADABLE_STATES = frozenset({"finished", "failed", "crashed", "killed", "preempted"})
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class RunReference:
    """Canonical coordinates that the W&B Public API uses to retrieve a run.

    Attributes:
        entity: User or team that owns the W&B project.
        project: W&B project containing the run.
        run_id: Immutable W&B run identifier, not its editable display name.
    """

    entity: str
    project: str
    run_id: str

    @property
    def path(self) -> str:
        """Return the ``entity/project/run-id`` path accepted by ``Api.run``."""
        return f"{self.entity}/{self.project}/{self.run_id}"


class RunSnapshot(BaseModel):
    """Stable JSON metadata stored as ``run.json`` inside every archive."""

    model_config = ConfigDict(extra="forbid")

    entity: str
    project: str
    run_id: str
    name: str | None
    url: str
    state: str
    created_at: str | None
    tags: list[str]
    notes: str | None
    config: dict[str, Any]
    summary: dict[str, Any]
    metadata: dict[str, Any] | None


class ArchivedFile(BaseModel):
    """Integrity metadata for one ZIP member other than ``manifest.json``."""

    model_config = ConfigDict(extra="forbid")

    path: str
    size_bytes: int
    sha256: str


class ArchiveManifest(BaseModel):
    """Schema and integrity metadata stored as ``manifest.json`` in a ZIP."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    archive_tag: str
    source_url: str
    files: list[ArchivedFile]


def _safe_component(value: str, label: str) -> str:
    """Reject W&B identifiers that could escape the repository archive tree.

    Args:
        value: One entity, project, or run-ID path component.
        label: Human-readable component name used in validation errors.

    Returns:
        ``value`` unchanged when it is safe to use as one local directory or
        filename component.

    Raises:
        ValueError: If the value is empty, special, or contains a path separator.
    """
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def parse_run_reference(
    reference: str,
    *,
    project: str | None = None,
    entity: str | None = None,
    default_entity: str | None = None,
) -> RunReference:
    """Normalize a W&B URL, canonical path, or bare ID for the mark command.

    Args:
        reference: A ``wandb.ai/<entity>/<project>/runs/<id>`` URL,
            ``entity/project/id`` path, or bare run ID.
        project: Project used only for a bare run ID. It is deliberately not a
            global default because this repository can contain multiple W&B
            projects.
        entity: Entity used only for a bare run ID. If omitted, the authenticated
            W&B default entity is used.
        default_entity: Authenticated default supplied by the W&B API. Callers
            pass it separately so fully qualified references do not require an
            otherwise unnecessary viewer request.

    Returns:
        Canonical entity, project, and immutable run ID coordinates.

    Raises:
        ValueError: If the format is ambiguous, incomplete, or unsafe.
    """
    parsed_url = urlparse(reference)
    if parsed_url.scheme:
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError(f"unsupported W&B run URL: {reference}")
        parts = [unquote(part) for part in parsed_url.path.split("/") if part]
        if len(parts) != 4 or parts[2] != "runs":
            raise ValueError("W&B run URLs must end with /<entity>/<project>/runs/<run-id>")
        resolved_entity, resolved_project, _, run_id = parts
    else:
        parts = [part for part in reference.strip("/").split("/") if part]
        if len(parts) == 4 and parts[2] == "runs":
            resolved_entity, resolved_project, _, run_id = parts
        elif len(parts) == 3:
            resolved_entity, resolved_project, run_id = parts
        elif len(parts) == 1:
            if project is None:
                raise ValueError("a bare run ID requires --project")
            resolved_entity = entity or default_entity or ""
            resolved_project = project
            run_id = parts[0]
        else:
            raise ValueError("run reference must be a W&B run URL, entity/project/run-id, or bare run ID")

    return RunReference(
        entity=_safe_component(resolved_entity, "entity"),
        project=_safe_component(resolved_project, "project"),
        run_id=_safe_component(run_id, "run ID"),
    )


def _json_bytes(value: Any) -> bytes:
    """Serialize W&B values as stable UTF-8 JSON, including NumPy-like values."""
    friendly_value = wandb.util.json_friendly_val(value)
    return (json.dumps(friendly_value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def _json_line_bytes(value: Any) -> bytes:
    """Serialize one W&B history row as a stable single-line JSON record."""
    friendly_value = wandb.util.json_friendly_val(value)
    return (json.dumps(friendly_value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))


def _validate_remote_file_name(name: str) -> PurePosixPath:
    """Validate one server-provided filename before the SDK writes it locally."""
    logical_path = PurePosixPath(name)
    if logical_path.is_absolute() or not logical_path.parts or any(part in {"", ".", ".."} for part in logical_path.parts):
        raise ValueError(f"unsafe W&B run filename: {name!r}")
    return logical_path


def _directory_size(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _check_size(size_bytes: int, description: str) -> None:
    if size_bytes > MAX_ARCHIVE_BYTES:
        raise ValueError(f"{description} is {size_bytes} bytes; the Git-safe limit is {MAX_ARCHIVE_BYTES} bytes")


def _write_run_contents(run: Any, contents_directory: Path) -> None:
    """Download one marked run into the temporary, unpacked archive tree.

    Args:
        run: W&B Public API run. It must expose entity, project, id, name, url,
            state, created_at, tags, notes, config, summary, metadata, files(),
            and scan_history(). File objects must expose name, size, and
            download(root, replace).
        contents_directory: Empty staging directory on the same filesystem as
            the final archive. The function writes ``run.json``, full
            ``history.jsonl``, downloaded run files below ``files/``, and a
            manifest enumerating the SHA-256 and byte size of every other member.

    Returns:
        ``None``. The caller zips and atomically publishes the populated tree.

    Invariants:
        The exact archive tag is checked again here even though discovery also
        filters by it. W&B artifacts are a separate API and are intentionally not
        downloaded. Size checks apply to both advertised remote sizes and actual
        staged bytes so an oversized run never reaches the Git-visible path.
    """
    if ARCHIVE_TAG not in set(run.tags or []):
        raise ValueError(f"refusing to archive unmarked run {run.entity}/{run.project}/{run.id}")

    remote_files = sorted(run.files(), key=lambda remote_file: remote_file.name)
    advertised_bytes = sum(int(remote_file.size or 0) for remote_file in remote_files)
    _check_size(advertised_bytes, "advertised W&B run files")

    run_snapshot = RunSnapshot(
        entity=run.entity,
        project=run.project,
        run_id=run.id,
        name=run.name,
        url=run.url,
        state=run.state,
        created_at=getattr(run, "created_at", None),
        tags=list(run.tags or []),
        notes=getattr(run, "notes", None),
        config=dict(run.config),
        summary=dict(run.summary),
        metadata=run.metadata,
    )
    _write_json(contents_directory / "run.json", run_snapshot.model_dump())

    history_path = contents_directory / "history.jsonl"
    with history_path.open("wb") as history_file:
        for row in run.scan_history(page_size=1000):
            history_file.write(_json_line_bytes(row))
            _check_size(history_file.tell(), "W&B run history")

    files_directory = contents_directory / "files"
    files_directory.mkdir()
    for remote_file in remote_files:
        logical_path = _validate_remote_file_name(remote_file.name)
        downloaded_file = remote_file.download(root=str(files_directory), replace=True)
        downloaded_file.close()
        downloaded_path = files_directory.joinpath(*logical_path.parts)
        if not downloaded_path.is_file() or downloaded_path.is_symlink():
            raise ValueError(f"W&B file did not produce a regular staged file: {remote_file.name!r}")
        _check_size(_directory_size(contents_directory), "staged W&B run")

    archived_files = [
        ArchivedFile(
            path=path.relative_to(contents_directory).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(contents_directory.rglob("*"))
        if path.is_file()
    ]
    manifest = ArchiveManifest(schema_version=1, archive_tag=ARCHIVE_TAG, source_url=run.url, files=archived_files)
    _write_json(contents_directory / "manifest.json", manifest.model_dump())
    _check_size(_directory_size(contents_directory), "staged W&B run")


def _write_deterministic_zip(contents_directory: Path, archive_path: Path) -> None:
    """Create a byte-stable ZIP by fixing member order, timestamps, and modes.

    Args:
        contents_directory: Staged tree whose regular files become ZIP members.
        archive_path: New ZIP destination within the per-run staging directory.

    Returns:
        ``None``. Callers validate the resulting size before publishing it.
    """
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source_path in sorted(contents_directory.rglob("*")):
            if not source_path.is_file():
                continue
            member_name = source_path.relative_to(contents_directory).as_posix()
            member = zipfile.ZipInfo(member_name, date_time=ZIP_TIMESTAMP)
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = (stat.S_IFREG | 0o644) << 16
            with source_path.open("rb") as source_file, archive.open(member, mode="w") as destination_file:
                shutil.copyfileobj(source_file, destination_file)


def archive_path_for_run(run: Any, archive_root: Path | None = None) -> Path:
    """Return ``<root>/<entity>/<project>/<run-id>.zip`` for a W&B run.

    Args:
        run: W&B Public API run exposing string ``entity``, ``project``, and
            immutable ``id`` attributes.
        archive_root: Storage root, defaulting to the repository's
            ``wandb_runs`` directory.

    Returns:
        The run's sole Git-visible archive path. Unsafe path components raise
        ``ValueError`` instead of being transformed into ambiguous names.
    """
    archive_root = archive_root or ARCHIVE_ROOT
    entity = _safe_component(run.entity, "entity")
    project = _safe_component(run.project, "project")
    run_id = _safe_component(run.id, "run ID")
    return archive_root / entity / project / f"{run_id}.zip"


def archive_marked_run(run: Any, archive_root: Path | None = None) -> tuple[Path, bool]:
    """Idempotently stage and atomically publish one marked W&B run.

    Args:
        run: W&B Public API run satisfying the contract documented by
            ``_write_run_contents``.
        archive_root: Repository-relative storage root. Production callers use
            ``wandb_runs``; tests inject a pytest-owned directory.

    Returns:
        ``(archive_path, created)``. ``created`` is ``False`` when the complete
        final path already existed before any network files were downloaded.

    Invariants:
        All temporary paths live below ``archive_root/.staging`` so
        ``os.replace`` publishes on one filesystem. A failure or interruption
        before that operation cannot create a partial final ZIP.
    """
    archive_root = archive_root or ARCHIVE_ROOT
    destination_path = archive_path_for_run(run, archive_root)
    if destination_path.exists():
        return destination_path, False

    staging_root = archive_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{run.id}-", dir=staging_root) as temporary_directory:
        temporary_path = Path(temporary_directory)
        contents_directory = temporary_path / "contents"
        contents_directory.mkdir()
        _write_run_contents(run, contents_directory)
        staged_archive_path = temporary_path / destination_path.name
        _write_deterministic_zip(contents_directory, staged_archive_path)
        staged_archive_path.chmod(0o644)
        _check_size(staged_archive_path.stat().st_size, "compressed W&B run archive")
        if destination_path.exists():
            return destination_path, False
        os.replace(staged_archive_path, destination_path)
    return destination_path, True


def _marked_runs(api: Any, entities: Iterable[str]) -> list[Any]:
    """Discover exact-tagged runs across every project in selected entities.

    Args:
        api: W&B Public API exposing ``projects(entity=...)`` and
            ``runs(entity/project, filters=...)``.
        entities: Entity names whose projects should be searched. Duplicate
            entity inputs are scanned only once.

    Returns:
        W&B Public API run objects carrying ``ARCHIVE_TAG``, deduplicated by
        ``entity/project/id`` and sorted by that canonical path. The local tag
        check defends against a server or fake returning a nonmatching run.
    """
    runs_by_path: dict[str, Any] = {}
    for entity in sorted(set(entities)):
        for project in api.projects(entity=entity):
            for run in api.runs(f"{entity}/{project.name}", filters={"tags": {"$in": [ARCHIVE_TAG]}}):
                if ARCHIVE_TAG in set(run.tags or []):
                    runs_by_path[f"{run.entity}/{run.project}/{run.id}"] = run
    return [runs_by_path[path] for path in sorted(runs_by_path)]


@click.group()
def cli() -> None:
    """Manage Git archives of explicitly marked W&B runs."""


@cli.command()
@click.argument("run_reference")
@click.option("--project", help="W&B project; required when RUN_REFERENCE is a bare run ID.")
@click.option("--entity", help="W&B entity for a bare run ID; defaults to the authenticated entity.")
def mark(run_reference: str, project: str | None, entity: str | None) -> None:
    """Mark RUN_REFERENCE for the next download operation."""
    api = wandb.Api()
    needs_default_entity = "/" not in run_reference and not urlparse(run_reference).scheme and entity is None
    default_entity = api.default_entity if needs_default_entity else None
    try:
        parsed_reference = parse_run_reference(
            run_reference,
            project=project,
            entity=entity,
            default_entity=default_entity,
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    run = api.run(parsed_reference.path)
    existing_tags = list(run.tags or [])
    if ARCHIVE_TAG in existing_tags:
        click.echo(f"Already marked {parsed_reference.path}")
        return
    run.tags = [*existing_tags, ARCHIVE_TAG]
    run.update()
    click.echo(f"Marked {parsed_reference.path}")


@cli.command()
@click.option("--entity", "entities", multiple=True, help="W&B entity to scan; repeat for multiple entities.")
def download(entities: tuple[str, ...]) -> None:
    """Archive every terminal, marked run that is not already present."""
    api = wandb.Api()
    selected_entities = entities or ((api.default_entity,) if api.default_entity else ())
    if not selected_entities:
        raise click.ClickException("no W&B entity supplied and the authenticated account has no default entity")

    created_count = 0
    existing_count = 0
    active_count = 0
    for run in _marked_runs(api, selected_entities):
        if str(run.state).lower() not in DOWNLOADABLE_STATES:
            active_count += 1
            click.echo(f"Skipping {run.entity}/{run.project}/{run.id}: state is {run.state}")
            continue
        try:
            archive_path, created = archive_marked_run(run)
        except ValueError as error:
            raise click.ClickException(str(error)) from error
        relative_archive_path = archive_path.relative_to(REPO_ROOT)
        if created:
            created_count += 1
            click.echo(f"Archived {run.entity}/{run.project}/{run.id} -> {relative_archive_path}")
        else:
            existing_count += 1
            click.echo(f"Already archived {run.entity}/{run.project}/{run.id}")

    click.echo(f"Done: {created_count} archived, {existing_count} already present, {active_count} still active")


if __name__ == "__main__":
    cli()
