"""Behavioral tests for archive loading and final-loss curve fits.

The test space partitions archive lookup into unique, missing, and duplicate
ZIPs; history extraction into preferred ``train/global_step``, ``_step``
fallback, missing metrics, and duplicate steps; chronological splits into
typical fractions, two-point series, and invalid fractions; power-law fits
into exact recoverable series and under-determined series; and sigmoid fits
into recoverable S-curves versus too-few-point skips.

The suite omits notebook plotting, IPython display, committed W&B ZIP bytes,
and numerical grid-search uniqueness for noisy sigmoid data.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wandb_runs.lib.archives import load_run_archive, metric_points, resolve_repo_and_archive_roots
from wandb_runs.lib.final_loss import (
    POWER_LAW_METRIC_KIND,
    annotate_sigmoid_catalog,
    build_power_law_catalog,
    endpoint_accuracy,
    fit_power_law,
    fit_sigmoid_loglog,
    predict_power_law,
    predict_sigmoid_loglog,
    run_power_law_sweep,
    runs_with_metric,
    split_chronological,
)


def _write_run_zip(archive_root: Path, entity: str, project: str, run_id: str, history_rows: list[dict[str, object]]) -> Path:
    destination = archive_root / entity / project / f"{run_id}.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr("run.json", json.dumps({"run_id": run_id, "state": "finished"}))
        archive.writestr("history.jsonl", "".join(json.dumps(row) + "\n" for row in history_rows))
    return destination


def test_resolve_roots_accepts_repo_root_or_wandb_runs_directory(tmp_path: Path) -> None:
    """Cover both supported working directories and reject an unrelated cwd."""
    repo_root = tmp_path / "repo"
    archive_root = repo_root / "wandb_runs"
    archive_root.mkdir(parents=True)

    assert resolve_repo_and_archive_roots(repo_root) == (repo_root, archive_root)
    assert resolve_repo_and_archive_roots(archive_root) == (repo_root, archive_root)
    with pytest.raises(RuntimeError, match="repository root or wandb_runs"):
        resolve_repo_and_archive_roots(tmp_path / "elsewhere")


def test_load_run_archive_requires_exactly_one_matching_zip(tmp_path: Path) -> None:
    """Cover unique success plus the empty-match and duplicate-match failures."""
    repo_root = tmp_path / "repo"
    archive_root = repo_root / "wandb_runs"
    history = [{"train/global_step": 1, "train/loss": 2.0}]
    _write_run_zip(archive_root, "entity-a", "project-a", "run1", history)

    loaded = load_run_archive("run1", archive_root=archive_root, repo_root=repo_root)
    assert loaded["archive_path"] == Path("wandb_runs/entity-a/project-a/run1.zip")
    assert loaded["snapshot"]["run_id"] == "run1"
    assert list(loaded["history"]["train/loss"]) == [2.0]

    with pytest.raises(RuntimeError, match="Expected one archive for missing"):
        load_run_archive("missing", archive_root=archive_root, repo_root=repo_root)

    _write_run_zip(archive_root, "entity-b", "project-b", "run1", history)
    with pytest.raises(RuntimeError, match="Expected one archive for run1"):
        load_run_archive("run1", archive_root=archive_root, repo_root=repo_root)


def test_metric_points_prefers_trainer_step_and_keeps_last_duplicate() -> None:
    """Cover preferred x-axis, _step fallback, missing metric, and duplicate steps."""
    preferred = pd.DataFrame(
        {
            "train/global_step": [1, 2, 2, 3],
            "_step": [10, 11, 12, 13],
            "train/loss": [4.0, 3.0, 2.5, None],
        }
    )
    points = metric_points(preferred, "train/loss")
    assert list(points["step"]) == [1.0, 2.0]
    assert list(points["value"]) == [4.0, 2.5]

    fallback = pd.DataFrame({"_step": [1, 2], "eval/loss": [5.0, 4.0]})
    assert list(metric_points(fallback, "eval/loss")["step"]) == [1.0, 2.0]
    assert metric_points(fallback, "train/loss").empty


def test_chronological_split_holds_out_the_latest_fraction() -> None:
    """Cover a typical 80/20 split, the two-point all-train partition, and an invalid fraction."""
    points = pd.DataFrame({"step": [1, 2, 3, 4, 5], "value": [5.0, 4.0, 3.0, 2.0, 1.0]})
    train_points, test_points = split_chronological(points, 0.8)
    assert list(train_points["step"]) == [1, 2, 3, 4]
    assert list(test_points["step"]) == [5]

    two = points.iloc[:2]
    train_two, test_two = split_chronological(two, 0.8)
    assert list(train_two["step"]) == [1, 2]
    assert test_two.empty

    with pytest.raises(ValueError, match="train_fraction"):
        split_chronological(points, 1.0)


def test_power_law_recovers_an_exact_series_and_zero_endpoint_error() -> None:
    """Cover closed-form recovery of A and beta and exact final/min prediction."""
    steps = pd.Series([1.0, 2.0, 4.0, 8.0])
    values = pd.Series(3.0 * steps.to_numpy() ** -0.5)
    fit = fit_power_law(steps, values)
    assert fit["A"] == pytest.approx(3.0)
    assert fit["beta"] == pytest.approx(-0.5)
    predicted = predict_power_law(steps, fit["A"], fit["beta"])
    assert predicted == pytest.approx(values.to_numpy())

    full_points = pd.DataFrame({"step": steps, "value": values})
    errors = endpoint_accuracy(full_points, fit["A"], fit["beta"])
    assert errors["rel_err_final"] == pytest.approx(0.0)
    assert errors["rel_err_min"] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="two distinct"):
        fit_power_law(pd.Series([2.0, 2.0]), pd.Series([1.0, 1.0]))


def test_sigmoid_recovers_plateaus_and_skips_too_few_points() -> None:
    """Cover an identifiable decreasing S-curve versus the four-point minimum."""
    steps = np.geomspace(1.0, 100.0, 40)
    values = predict_sigmoid_loglog(steps, y_hi=1.5, y_lo=-0.5, t0=10.0, k=3.0)
    fit = fit_sigmoid_loglog(pd.Series(steps), pd.Series(values))
    assert fit["status"] == "ok"
    assert fit["y_hi"] == pytest.approx(1.5, abs=0.05)
    assert fit["y_lo"] == pytest.approx(-0.5, abs=0.05)
    assert fit["t0"] == pytest.approx(10.0, rel=0.3)

    skipped = fit_sigmoid_loglog(pd.Series([1.0, 2.0, 3.0]), pd.Series([3.0, 2.0, 1.0]))
    assert skipped["status"].startswith("need at least 4")
    assert np.isnan(skipped["t0"])


def test_catalog_and_sweep_score_the_same_series_final_loss() -> None:
    """Cover eligible scopes, cartesian size, and a one-row power-law sweep."""
    long_steps = np.arange(1, 21, dtype=float)
    short_steps = np.arange(1, 6, dtype=float)
    runs_by_id = {
        "run-a": {"history": pd.DataFrame({"train/global_step": long_steps, "train/loss": 2.0 * long_steps**-0.4})},
        "run-b": {"history": pd.DataFrame({"train/global_step": short_steps, "train/loss": 2.0 * short_steps**-0.4})},
    }
    run_ids = ("run-a", "run-b")
    metrics = ("train/loss", "eval/loss")
    kinds = {metric: POWER_LAW_METRIC_KIND.get(metric, "total") for metric in metrics}

    assert runs_with_metric("train/loss", runs_by_id, run_ids) == ["run-a", "run-b"]
    assert runs_with_metric("eval/loss", runs_by_id, run_ids) == []

    catalog = build_power_law_catalog(runs_by_id, run_ids, metrics, kinds, (0.5,))
    assert set(catalog["scope"]) == {"run-a", "run-b", "joint"}
    assert set(catalog["metric"]) == {"train/loss"}
    assert len(catalog) == 3

    sigmoid_catalog = annotate_sigmoid_catalog(catalog, runs_by_id, run_ids)
    assert bool(sigmoid_catalog.loc[sigmoid_catalog["scope"] == "run-a", "sigmoid_eligible"].iloc[0]) is True

    fits, endpoints = run_power_law_sweep(catalog, runs_by_id, run_ids)
    run_a_endpoint = endpoints.loc[(endpoints["scope"] == "run-a") & (endpoints["target_run"] == "run-a")].iloc[0]
    assert run_a_endpoint["rel_err_final"] == pytest.approx(0.0, abs=1e-6)
    assert "eval/" not in "".join(fits["metric"])
