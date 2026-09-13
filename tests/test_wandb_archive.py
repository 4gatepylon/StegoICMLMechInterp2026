"""Behavioral tests for repository-wide W&B run archiving.

The test space partitions references into URL, canonical-path, and bare-ID
forms; tags into newly added, already present, and absent; downloads into new,
existing, active, oversized, and interrupted runs; and archive construction
into repeated builds with stable bytes. W&B and its remote file transport are
replaced with fakes, while ZIP, hashing, staging, and atomic local publication
use real pytest-owned files.

The suite omits real W&B authentication and network behavior, operating-system
guarantees under concurrent downloaders or power loss, W&B artifact downloads,
archives above the Git-safe limit, and performance with very large histories.
"""

from __future__ import annotations

import json
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import wandb_archive
from wandb_archive import ARCHIVE_TAG, MAX_ARCHIVE_BYTES, archive_marked_run, parse_run_reference


@dataclass(frozen=True)
class FakeProject:
    name: str


class FakeRemoteFile:
    """Provide the run-file subset consumed by the production archiver."""

    def __init__(self, name: str, contents: bytes, *, advertised_size: int | None = None, fail: bool = False) -> None:
        self.name = name
        self.contents = contents
        self.size = len(contents) if advertised_size is None else advertised_size
        self.fail = fail
        self.download_count = 0

    def download(self, root: str, replace: bool) -> Any:
        assert replace is True
        self.download_count += 1
        destination = Path(root) / self.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.contents)
        if self.fail:
            raise RuntimeError("simulated interrupted download")
        return destination.open()


class FakeRun:
    """Provide the Public API run fields consumed across mark and archive."""

    def __init__(
        self,
        run_id: str,
        *,
        tags: list[str] | None = None,
        state: str = "finished",
        remote_files: list[FakeRemoteFile] | None = None,
    ) -> None:
        self.entity = "test-entity"
        self.project = "test-project"
        self.id = run_id
        self.name = f"display-{run_id}"
        self.url = f"https://wandb.ai/{self.entity}/{self.project}/runs/{run_id}"
        self.state = state
        self.created_at = "2026-09-13T12:00:00Z"
        self.tags = list(tags or [])
        self.notes = "test run"
        self.config = {"learning_rate": 0.001}
        self.summary = {"loss": 0.25}
        self.metadata = {"git": {"commit": "abc123"}}
        self.remote_files = list(remote_files or [FakeRemoteFile("output.log", b"training complete\n")])
        self.history = [{"_step": 0, "loss": 1.0}, {"_step": 1, "loss": 0.25}]
        self.update_count = 0

    def update(self) -> None:
        self.update_count += 1

    def files(self) -> list[FakeRemoteFile]:
        return self.remote_files

    def scan_history(self, page_size: int) -> list[dict[str, float]]:
        assert page_size == 1000
        return self.history


class FakeApi:
    """Record project/run queries and return configured fake W&B objects."""

    def __init__(self, runs: list[FakeRun]) -> None:
        self.default_entity = "test-entity"
        self.configured_runs = runs
        self.run_paths: list[str] = []
        self.run_queries: list[tuple[str, dict[str, object]]] = []

    def run(self, path: str) -> FakeRun:
        self.run_paths.append(path)
        return self.configured_runs[0]

    def projects(self, entity: str) -> list[FakeProject]:
        assert entity == "test-entity"
        return [FakeProject("test-project")]

    def runs(self, path: str, filters: dict[str, object]) -> list[FakeRun]:
        self.run_queries.append((path, filters))
        return self.configured_runs


@pytest.mark.parametrize(
    ("reference", "project", "entity"),
    [
        ("https://wandb.ai/test-entity/test-project/runs/run123", None, None),
        ("test-entity/test-project/run123", None, None),
        ("run123", "test-project", "test-entity"),
    ],
)
def test_run_reference_forms_resolve_to_one_canonical_path(reference: str, project: str | None, entity: str | None) -> None:
    """Cover every supported reference partition; self-hosted URL conventions are omitted."""
    parsed = parse_run_reference(reference, project=project, entity=entity)

    assert parsed.path == "test-entity/test-project/run123"


def test_bare_run_id_requires_a_project() -> None:
    """Cover the ambiguity boundary that prevents repository-wide implicit project selection."""
    with pytest.raises(ValueError, match="requires --project"):
        parse_run_reference("run123", default_entity="test-entity")


def test_mark_preserves_tags_and_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover newly marked and already marked calls without contacting W&B."""
    run = FakeRun("run123", tags=["baseline"])
    api = FakeApi([run])
    monkeypatch.setattr(wandb_archive.wandb, "Api", lambda: api)
    runner = CliRunner()

    first_result = runner.invoke(wandb_archive.cli, ["mark", run.url])
    second_result = runner.invoke(wandb_archive.cli, ["mark", run.url])

    assert first_result.exit_code == 0
    assert second_result.exit_code == 0
    assert run.tags == ["baseline", ARCHIVE_TAG]
    assert run.update_count == 1
    assert api.run_paths == ["test-entity/test-project/run123"] * 2


def test_marked_run_discovery_uses_exact_filter_and_defends_against_unmarked_results() -> None:
    """Cover tagged and untagged server results; W&B's filter implementation is mocked."""
    marked_run = FakeRun("marked", tags=[ARCHIVE_TAG])
    unmarked_run = FakeRun("unmarked", tags=["other"])
    api = FakeApi([unmarked_run, marked_run])

    discovered_runs = wandb_archive._marked_runs(api, ["test-entity"])

    assert discovered_runs == [marked_run]
    assert api.run_queries == [
        ("test-entity/test-project", {"tags": {"$in": [ARCHIVE_TAG]}}),
    ]


def test_archive_is_complete_deterministic_and_idempotent(tmp_path: Path) -> None:
    """Cover real ZIP creation twice plus the existing-destination no-download partition."""
    remote_file = FakeRemoteFile("media/example.txt", b"example payload\n")
    run = FakeRun("run123", tags=["baseline", ARCHIVE_TAG], remote_files=[remote_file])
    first_root = tmp_path / "first" / "wandb_runs"
    second_root = tmp_path / "second" / "wandb_runs"

    first_archive, first_created = archive_marked_run(run, first_root)
    downloads_after_first_build = remote_file.download_count
    existing_archive, existing_created = archive_marked_run(run, first_root)
    second_archive, second_created = archive_marked_run(run, second_root)

    assert first_created is True
    assert existing_created is False
    assert second_created is True
    assert existing_archive == first_archive
    assert remote_file.download_count == downloads_after_first_build + 1
    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert stat.S_IMODE(first_archive.stat().st_mode) == 0o644

    with zipfile.ZipFile(first_archive) as archive:
        assert archive.namelist() == ["files/media/example.txt", "history.jsonl", "manifest.json", "run.json"]
        snapshot = json.loads(archive.read("run.json"))
        history = [json.loads(line) for line in archive.read("history.jsonl").splitlines()]
        manifest = json.loads(archive.read("manifest.json"))
    assert snapshot["run_id"] == "run123"
    assert history == run.history
    assert {entry["path"] for entry in manifest["files"]} == {"files/media/example.txt", "history.jsonl", "run.json"}


def test_interrupted_download_never_publishes_partial_archive(tmp_path: Path) -> None:
    """Cover failure after a partial staged file; process-kill cleanup is omitted."""
    run = FakeRun("interrupted", tags=[ARCHIVE_TAG], remote_files=[FakeRemoteFile("output.log", b"partial", fail=True)])
    archive_root = tmp_path / "wandb_runs"

    with pytest.raises(RuntimeError, match="simulated interrupted download"):
        archive_marked_run(run, archive_root)

    assert not (archive_root / run.entity / run.project / f"{run.id}.zip").exists()
    assert list((archive_root / ".staging").iterdir()) == []


def test_unmarked_and_oversized_runs_are_rejected_before_download(tmp_path: Path) -> None:
    """Cover authorization and advertised-size guards; compressed-size overflow is omitted."""
    ordinary_file = FakeRemoteFile("output.log", b"ordinary")
    unmarked_run = FakeRun("unmarked", remote_files=[ordinary_file])
    with pytest.raises(ValueError, match="refusing to archive unmarked"):
        archive_marked_run(unmarked_run, tmp_path / "unmarked")
    assert ordinary_file.download_count == 0

    oversized_file = FakeRemoteFile("large.bin", b"small fake", advertised_size=MAX_ARCHIVE_BYTES + 1)
    oversized_run = FakeRun("oversized", tags=[ARCHIVE_TAG], remote_files=[oversized_file])
    with pytest.raises(ValueError, match="Git-safe limit"):
        archive_marked_run(oversized_run, tmp_path / "oversized")
    assert oversized_file.download_count == 0


def test_download_skips_active_runs_and_archives_only_completed_marked_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover active and terminal state partitions through the public Click command."""
    active_run = FakeRun("active", tags=[ARCHIVE_TAG], state="running")
    completed_run = FakeRun("completed", tags=[ARCHIVE_TAG])
    api = FakeApi([active_run, completed_run])
    monkeypatch.setattr(wandb_archive.wandb, "Api", lambda: api)
    monkeypatch.setattr(wandb_archive, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(wandb_archive, "ARCHIVE_ROOT", tmp_path / "wandb_runs")
    runner = CliRunner()

    result = runner.invoke(wandb_archive.cli, ["download", "--entity", "test-entity"])

    assert result.exit_code == 0, result.output
    assert "state is running" in result.output
    assert not (tmp_path / "wandb_runs/test-entity/test-project/active.zip").exists()
    assert (tmp_path / "wandb_runs/test-entity/test-project/completed.zip").is_file()
