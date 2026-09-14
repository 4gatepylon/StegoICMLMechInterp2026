"""Load committed W&B run ZIP archives and extract numeric history series.

The inspection notebook is the user-facing viewer. This module exists so archive
IO can be imported, tested, and reviewed independently of plotting cells.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd


def resolve_repo_and_archive_roots(working_directory: Path) -> tuple[Path, Path]:
    """Resolve the repository root and ``wandb_runs`` archive root from ``cwd``.

    Args:
        working_directory: The process working directory, typically ``Path.cwd()``.
            Callers may pass either the repository root or the ``wandb_runs``
            directory. Other locations are rejected so an archive glob cannot
            silently search the wrong tree.

    Returns:
        ``(repo_root, archive_root)``. ``archive_root`` is always
        ``repo_root / "wandb_runs"``. ``load_run_archive`` uses ``archive_root``
        to find ZIPs and ``repo_root`` to relativize the returned path.

    Raises:
        RuntimeError: If ``working_directory`` is neither the repository root
            nor the ``wandb_runs`` directory.
    """
    working_directory = working_directory.resolve()
    if (working_directory / "wandb_runs").is_dir():
        repo_root = working_directory
    elif working_directory.name == "wandb_runs" and (working_directory.parent / "wandb_runs").is_dir():
        repo_root = working_directory.parent
    else:
        raise RuntimeError("Run this notebook from the repository root or wandb_runs directory.")
    return repo_root, repo_root / "wandb_runs"


def load_run_archive(run_id: str, *, archive_root: Path, repo_root: Path) -> dict[str, Any]:
    """Load one uniquely identified run snapshot and its full history.

    Args:
        run_id: Immutable W&B run ID used as the archive filename stem.
        archive_root: Directory that contains ``<entity>/<project>/<run-id>.zip``.
            Production callers pass the repository ``wandb_runs`` directory.
        repo_root: Repository root used only to relativize ``archive_path``.

    Returns:
        A dictionary with ``archive_path`` (repository-relative ``Path``),
        ``snapshot`` (the complete ``run.json`` mapping), and ``history`` (a
        ``DataFrame`` containing every JSONL history row and metric column).
        Plotting cells consume ``snapshot`` for labels and ``history`` for data.

    Raises:
        RuntimeError: If no archive or more than one archive has the ID.
    """
    matching_paths = sorted(archive_root.glob(f"*/*/{run_id}.zip"))
    if len(matching_paths) != 1:
        raise RuntimeError(f"Expected one archive for {run_id}, found {matching_paths}")
    archive_path = matching_paths[0]
    with ZipFile(archive_path) as archive:
        snapshot = json.loads(archive.read("run.json"))
        history_rows = [json.loads(line) for line in archive.read("history.jsonl").splitlines()]
    return {
        "archive_path": archive_path.relative_to(repo_root),
        "snapshot": snapshot,
        "history": pd.DataFrame(history_rows),
    }


def metric_points(history: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return numeric ``(step, value)`` observations for one metric.

    ``train/global_step`` is the preferred x-coordinate because W&B ``_step``
    also advances for evaluation log events. ``_step`` is used only when the
    trainer-specific coordinate is absent. Duplicate steps retain the final
    logged value. Missing metrics return an empty two-column DataFrame.
    """
    step_column = "train/global_step" if "train/global_step" in history else "_step"
    if metric not in history or step_column not in history:
        return pd.DataFrame(columns=["step", "value"])
    points = history.loc[history[metric].notna(), [step_column, metric]].copy()
    points.columns = ["step", "value"]
    points["step"] = pd.to_numeric(points["step"], errors="coerce")
    points["value"] = pd.to_numeric(points["value"], errors="coerce")
    return points.dropna().drop_duplicates("step", keep="last").sort_values("step")
