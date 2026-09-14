"""Predict a logged series' final and minimum loss from an early-step prefix.

The inspection notebook remains the visualization front end. This module holds
the closed-form power-law and log-log sigmoid fits, the chronological split,
and the cartesian experiment sweep so those pieces can be reviewed and tested
without executing plotting cells.

Callers that previously closed over notebook globals now pass ``runs_by_id``,
``run_ids``, ``metrics``, ``metric_kinds``, and ``train_fractions`` explicitly.
The numeric procedures are otherwise the ones used by the notebook.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from wandb_runs.lib.archives import metric_points

POWER_LAW_METRICS = (
    "train/loss",
    "eval/loss",
    "train/prefix_loss",
    "train/data_loss",
    "eval/prefix_loss",
    "eval/data_loss",
)
POWER_LAW_METRIC_KIND = {
    "train/loss": "total",
    "eval/loss": "total",
    "train/prefix_loss": "prefix NLL",
    "train/data_loss": "data KL",
    "eval/prefix_loss": "prefix NLL",
    "eval/data_loss": "data KL",
}
SIGMOID_MIN_TRAIN_POINTS = 4
SIGMOID_K_GRID = np.geomspace(0.25, 12.0, 16)
SIGMOID_T0_GRID_SIZE = 21

RunArchiveMap = Mapping[str, Mapping[str, Any]]


def positive_metric_points(history: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Drop non-positive steps or losses so ``log`` is defined for the closed-form fit."""
    points = metric_points(history, metric)
    return points.loc[(points["step"] > 0) & (points["value"] > 0)].copy()


def runs_with_metric(metric: str, runs_by_id: RunArchiveMap, run_ids: Sequence[str]) -> list[str]:
    """Return run IDs whose archive has at least one positive ``metric`` point.

    Args:
        metric: Exact ``history.jsonl`` column name.
        runs_by_id: Mapping from run ID to an archive dict whose ``history``
            key is the ``DataFrame`` produced by ``load_run_archive``.
        run_ids: Scan order. Callers pass the notebook's ``RUN_IDS``.
    """
    return [run_id for run_id in run_ids if not positive_metric_points(runs_by_id[run_id]["history"], metric).empty]


def split_chronological(points: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one series by row order after sorting on step.

    ``train_fraction`` is the earlier share of rows used to fit. The remainder
    is the extrapolation test set. Two or fewer points are used entirely for
    the fit because a log-log line needs two distinct steps.
    """
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    ordered = points.sort_values("step", kind="mergesort")
    if len(ordered) <= 2:
        return ordered.copy(), ordered.iloc[0:0].copy()
    n_train = min(max(2, int(len(ordered) * train_fraction)), len(ordered) - 1)
    return ordered.iloc[:n_train].copy(), ordered.iloc[n_train:].copy()


def fit_power_law(steps: pd.Series, values: pd.Series) -> dict[str, float]:
    """Return ``A`` and ``beta`` for ``value = A * step ** beta``.

    The fit is ordinary least squares of ``log value`` on ``log step``.

    Returns:
        ``A``: positive scale. ``beta``: log-log slope. ``n``: training count.
        Sweep tables and overlay curves consume these keys.
    """
    log_x = np.log(np.asarray(steps, dtype=float))
    log_y = np.log(np.asarray(values, dtype=float))
    if log_x.size < 2 or np.unique(log_x).size < 2:
        raise ValueError("power-law fit needs at least two distinct positive steps")
    x_centered = log_x - log_x.mean()
    beta = float(np.dot(x_centered, log_y - log_y.mean()) / np.dot(x_centered, x_centered))
    log_a = float(log_y.mean() - beta * log_x.mean())
    return {"A": float(np.exp(log_a)), "beta": beta, "n": float(log_x.size)}


def predict_power_law(steps: pd.Series | np.ndarray, A: float, beta: float) -> np.ndarray:
    """Evaluate ``A * step ** beta`` on the given optimizer steps."""
    return np.asarray(A, dtype=float) * np.asarray(steps, dtype=float) ** beta


def power_law_errors(points: pd.DataFrame, A: float, beta: float) -> dict[str, float]:
    """MAE and log-space MSE. Empty frames yield NaNs so the table stays rectangular."""
    if points.empty:
        return {"mae": float("nan"), "log_mse": float("nan")}
    predicted = predict_power_law(points["step"], A, beta)
    observed = points["value"].to_numpy(dtype=float)
    return {
        "mae": float(np.mean(np.abs(observed - predicted))),
        "log_mse": float(np.mean((np.log(observed) - np.log(predicted)) ** 2)),
    }


def endpoint_accuracy(full_points: pd.DataFrame, A: float, beta: float) -> dict[str, float]:
    """Score the power law's estimate of the full curve's last and minimum loss.

    ``predicted_final`` is ``A * step_last ** beta``. ``predicted_min`` is the
    minimum of that same curve on the observed steps. Both are compared to the
    actual last value and the actual minimum of the full series.
    """
    ordered = full_points.sort_values("step", kind="mergesort")
    final_step = float(ordered["step"].iloc[-1])
    actual_final = float(ordered["value"].iloc[-1])
    actual_min = float(ordered["value"].min())
    predicted = predict_power_law(ordered["step"], A, beta)
    predicted_final = float(predicted[-1])
    predicted_min = float(predicted.min())
    return {
        "final_step": final_step,
        "actual_final": actual_final,
        "predicted_final": predicted_final,
        "abs_err_final": abs(predicted_final - actual_final),
        "rel_err_final": abs(predicted_final - actual_final) / actual_final,
        "actual_min": actual_min,
        "predicted_min": predicted_min,
        "abs_err_min": abs(predicted_min - actual_min),
        "rel_err_min": abs(predicted_min - actual_min) / actual_min,
    }


def power_law_scopes(metric: str, runs_by_id: RunArchiveMap, run_ids: Sequence[str]) -> list[str]:
    """Per-run scopes that have ``metric``, plus ``joint`` when at least two runs do."""
    available = runs_with_metric(metric, runs_by_id, run_ids)
    if len(available) >= 2:
        return [*available, "joint"]
    return list(available)


def describe_train_set(train_fraction: float, metric: str, scope: str, available: list[str]) -> str:
    """One-line train-set description for the catalog and result tables."""
    if scope == "joint":
        members = ", ".join(available)
        return f"pooled earliest {train_fraction:.0%} of {members} {metric}"
    return f"earliest {train_fraction:.0%} of {scope} {metric}"


def describe_eval_set(train_fraction: float, metric: str, scope: str, available: list[str]) -> str:
    """Primary score: predict that same series' full-curve final and minimum loss."""
    family = "eval" if metric.startswith("eval/") else "train"
    if scope == "joint":
        members = ", ".join(available)
        return (
            f"primary: each member's final/min {family} {metric}; "
            f"diagnostic: pooled latest {1.0 - train_fraction:.0%} tail of {members} {metric}"
        )
    return (
        f"primary: final/min {family} {metric} on {scope}; "
        f"diagnostic: latest {1.0 - train_fraction:.0%} tail of {scope} {metric}"
    )


def build_power_law_definitions(
    runs_by_id: RunArchiveMap,
    run_ids: Sequence[str],
    metrics: Sequence[str],
    metric_kinds: Mapping[str, str],
) -> pd.DataFrame:
    """Eligible (metric, scope) pairs. Fractions are applied later as a cartesian axis."""
    rows: list[dict[str, object]] = []
    for metric in metrics:
        available = runs_with_metric(metric, runs_by_id, run_ids)
        for scope in power_law_scopes(metric, runs_by_id, run_ids):
            rows.append(
                {
                    "metric": metric,
                    "kind": metric_kinds[metric],
                    "scope": scope,
                    "available_runs": ", ".join(available),
                    "n_runs": len(available),
                }
            )
    return pd.DataFrame(rows)


def build_skipped_power_law_pairs(
    runs_by_id: RunArchiveMap,
    run_ids: Sequence[str],
    metrics: Sequence[str],
    metric_kinds: Mapping[str, str],
) -> pd.DataFrame:
    """(metric, run) pairs with no positive points, so they are outside the cartesian grid."""
    rows: list[dict[str, object]] = []
    for metric in metrics:
        available = set(runs_with_metric(metric, runs_by_id, run_ids))
        for run_id in run_ids:
            if run_id not in available:
                rows.append(
                    {
                        "metric": metric,
                        "kind": metric_kinds[metric],
                        "scope": run_id,
                        "reason": "archive has no positive points for this series",
                    }
                )
        if len(available) < 2:
            rows.append(
                {
                    "metric": metric,
                    "kind": metric_kinds[metric],
                    "scope": "joint",
                    "reason": "joint needs at least two runs with this series",
                }
            )
    return pd.DataFrame(rows)


def build_power_law_catalog(
    runs_by_id: RunArchiveMap,
    run_ids: Sequence[str],
    metrics: Sequence[str],
    metric_kinds: Mapping[str, str],
    train_fractions: Sequence[float],
) -> pd.DataFrame:
    """Cartesian product of ``train_fractions`` × eligible (metric, scope)."""
    rows: list[dict[str, object]] = []
    for train_fraction in train_fractions:
        for metric in metrics:
            available = runs_with_metric(metric, runs_by_id, run_ids)
            for scope in power_law_scopes(metric, runs_by_id, run_ids):
                rows.append(
                    {
                        "train_fraction": train_fraction,
                        "metric": metric,
                        "kind": metric_kinds[metric],
                        "scope": scope,
                        "train_set": describe_train_set(train_fraction, metric, scope, available),
                        "eval_set": describe_eval_set(train_fraction, metric, scope, available),
                    }
                )
    return pd.DataFrame(rows)


def scope_series(
    metric: str,
    scope: str,
    train_fraction: float,
    runs_by_id: RunArchiveMap,
    run_ids: Sequence[str],
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Collect chronological splits and full curves for one (metric, scope)."""
    available = runs_with_metric(metric, runs_by_id, run_ids)
    members = available if scope == "joint" else [scope]
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    full_by_run: dict[str, pd.DataFrame] = {}
    for run_id in members:
        full_points = positive_metric_points(runs_by_id[run_id]["history"], metric)
        train_points, test_points = split_chronological(full_points, train_fraction)
        train_parts.append(train_points.assign(run_id=run_id))
        test_parts.append(test_points.assign(run_id=run_id))
        full_by_run[run_id] = full_points
    return members, pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True), full_by_run


def run_power_law_sweep(catalog: pd.DataFrame, runs_by_id: RunArchiveMap, run_ids: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every catalog row and score tails plus each target run's final/min loss.

    Returns:
        ``fits``: one row per (train_fraction, metric, scope) with ``A``, ``beta``,
        tail MAE/log-MSE. ``endpoints``: one row per target run under that fit,
        with predicted vs actual final and minimum loss. Display cells consume both.
    """
    fit_rows: list[dict[str, object]] = []
    endpoint_rows: list[dict[str, object]] = []
    for experiment in catalog.itertuples(index=False):
        members, train_points, test_points, full_by_run = scope_series(
            experiment.metric, experiment.scope, float(experiment.train_fraction), runs_by_id, run_ids
        )
        fit = fit_power_law(train_points["step"], train_points["value"])
        train_errors = power_law_errors(train_points, fit["A"], fit["beta"])
        test_errors = power_law_errors(test_points, fit["A"], fit["beta"])
        fit_rows.append(
            {
                "train_fraction": experiment.train_fraction,
                "metric": experiment.metric,
                "kind": experiment.kind,
                "scope": experiment.scope,
                "train_set": experiment.train_set,
                "eval_set": experiment.eval_set,
                "n_train": int(fit["n"]),
                "n_test": len(test_points),
                "A": fit["A"],
                "beta": fit["beta"],
                "train_mae": train_errors["mae"],
                "test_mae": test_errors["mae"],
                "train_log_mse": train_errors["log_mse"],
                "test_log_mse": test_errors["log_mse"],
            }
        )
        for target_run, full_points in full_by_run.items():
            endpoint_rows.append(
                {
                    "train_fraction": experiment.train_fraction,
                    "metric": experiment.metric,
                    "kind": experiment.kind,
                    "scope": experiment.scope,
                    "target_run": target_run,
                    **endpoint_accuracy(full_points, fit["A"], fit["beta"]),
                }
            )
    return pd.DataFrame(fit_rows), pd.DataFrame(endpoint_rows)


def _sigmoid_weights(log_steps: np.ndarray, log_t0: float, k: float) -> np.ndarray:
    """Early-time weight ``z``: 1 on the high plateau, 0 on the low plateau."""
    return 1.0 / (1.0 + np.exp(np.clip(k * (log_steps - log_t0), -60.0, 60.0)))


def predict_sigmoid_loglog(steps: pd.Series | np.ndarray, y_hi: float, y_lo: float, t0: float, k: float) -> np.ndarray:
    """Evaluate the log-log S-curve ``loss = exp(y_lo + (y_hi - y_lo) * z(step))``.

    ``z`` is a logistic in ``log(step)``. Early steps sit near ``exp(y_hi)``, late
    steps near ``exp(y_lo)``, and the middle is approximately linear in log-log
    space with steepness ``k`` around midpoint step ``t0``.
    """
    log_steps = np.log(np.asarray(steps, dtype=float))
    weights = _sigmoid_weights(log_steps, float(np.log(t0)), float(k))
    return np.exp(y_lo + (y_hi - y_lo) * weights)


def _ols_sigmoid_plateaus(
    log_steps: np.ndarray, log_values: np.ndarray, log_t0: float, k: float
) -> tuple[float, float, float] | None:
    """Closed-form OLS for the two plateaus given a fixed midpoint and steepness.

    ``log value = y_lo + (y_hi - y_lo) * z``, so the design matrix is ``[1, z]``.
    Returns ``(y_hi, y_lo, sse)`` or ``None`` when ``z`` is degenerate.
    """
    weights = _sigmoid_weights(log_steps, log_t0, k)
    if float(weights.std()) < 1e-6:
        return None
    design = np.column_stack((np.ones(log_steps.size), weights))
    coefficients, _, rank, _ = np.linalg.lstsq(design, log_values, rcond=None)
    if rank < 2:
        return None
    y_lo = float(coefficients[0])
    y_hi = y_lo + float(coefficients[1])
    residual = log_values - design @ coefficients
    return y_hi, y_lo, float(np.dot(residual, residual))


def _search_sigmoid_grid(
    log_steps: np.ndarray, log_values: np.ndarray, log_t0_grid: np.ndarray, k_grid: np.ndarray
) -> tuple[float, float, float, float, float] | None:
    """Return ``(sse, y_hi, y_lo, log_t0, k)`` for the best identifiable grid cell."""
    best: tuple[float, float, float, float, float] | None = None
    for log_t0 in log_t0_grid:
        for k in k_grid:
            fitted = _ols_sigmoid_plateaus(log_steps, log_values, float(log_t0), float(k))
            if fitted is None:
                continue
            y_hi, y_lo, sse = fitted
            if best is None or sse < best[0]:
                best = (sse, y_hi, y_lo, float(log_t0), float(k))
    return best


def _empty_sigmoid_fit(n: int, status: str) -> dict[str, float | str]:
    return {
        "y_hi": float("nan"),
        "y_lo": float("nan"),
        "t0": float("nan"),
        "k": float("nan"),
        "n": float(n),
        "log_sse": float("nan"),
        "status": status,
    }


def fit_sigmoid_loglog(steps: pd.Series, values: pd.Series) -> dict[str, float | str]:
    """Fit the 4-parameter log-log S-curve.

    Nonlinear parameters ``t0`` and ``k`` are chosen by a coarse-then-fine grid.
    Plateaus ``y_hi`` and ``y_lo`` are closed-form OLS at each grid cell. Needs
    at least ``SIGMOID_MIN_TRAIN_POINTS`` distinct positive steps.

    Returns:
        ``y_hi``, ``y_lo``, ``t0``, ``k``, ``n``, ``log_sse``, ``status``.
        Sweep tables and overlay curves consume these keys. ``status`` is
        ``ok`` or a skip reason; skipped rows leave the numeric fields as NaN.
    """
    log_steps = np.log(np.asarray(steps, dtype=float))
    log_values = np.log(np.asarray(values, dtype=float))
    n_points = int(log_steps.size)
    if n_points < SIGMOID_MIN_TRAIN_POINTS or np.unique(log_steps).size < SIGMOID_MIN_TRAIN_POINTS:
        return _empty_sigmoid_fit(
            n_points,
            f"need at least {SIGMOID_MIN_TRAIN_POINTS} distinct train points for a 4-parameter log-log sigmoid",
        )
    log_t0_grid = np.linspace(float(log_steps.min()), float(log_steps.max()), SIGMOID_T0_GRID_SIZE)
    best = _search_sigmoid_grid(log_steps, log_values, log_t0_grid, SIGMOID_K_GRID)
    if best is None:
        return _empty_sigmoid_fit(n_points, "sigmoid grid search found no identifiable plateaus")
    sse, y_hi, y_lo, log_t0, k = best
    spacing = (float(log_steps.max()) - float(log_steps.min())) / max(SIGMOID_T0_GRID_SIZE - 1, 1)
    refined_t0 = np.linspace(
        max(float(log_steps.min()), log_t0 - 2 * spacing),
        min(float(log_steps.max()), log_t0 + 2 * spacing),
        11,
    )
    refined_k = np.geomspace(max(k / 3.0, 0.1), k * 3.0, 11)
    refined = _search_sigmoid_grid(log_steps, log_values, refined_t0, refined_k)
    if refined is not None:
        sse, y_hi, y_lo, log_t0, k = refined
    return {
        "y_hi": y_hi,
        "y_lo": y_lo,
        "t0": float(np.exp(log_t0)),
        "k": k,
        "n": float(n_points),
        "log_sse": sse,
        "status": "ok",
    }


def sigmoid_errors(points: pd.DataFrame, fit: dict[str, float | str]) -> dict[str, float]:
    """MAE and log-space MSE for a sigmoid fit. Empty or skipped fits yield NaNs."""
    if points.empty or fit["status"] != "ok":
        return {"mae": float("nan"), "log_mse": float("nan")}
    predicted = predict_sigmoid_loglog(points["step"], float(fit["y_hi"]), float(fit["y_lo"]), float(fit["t0"]), float(fit["k"]))
    observed = points["value"].to_numpy(dtype=float)
    return {
        "mae": float(np.mean(np.abs(observed - predicted))),
        "log_mse": float(np.mean((np.log(observed) - np.log(predicted)) ** 2)),
    }


def sigmoid_endpoint_accuracy(full_points: pd.DataFrame, fit: dict[str, float | str]) -> dict[str, float]:
    """Score the sigmoid's estimate of the full curve's last and minimum loss."""
    ordered = full_points.sort_values("step", kind="mergesort")
    final_step = float(ordered["step"].iloc[-1])
    actual_final = float(ordered["value"].iloc[-1])
    actual_min = float(ordered["value"].min())
    nan_block = {
        "final_step": final_step,
        "actual_final": actual_final,
        "predicted_final": float("nan"),
        "abs_err_final": float("nan"),
        "rel_err_final": float("nan"),
        "actual_min": actual_min,
        "predicted_min": float("nan"),
        "abs_err_min": float("nan"),
        "rel_err_min": float("nan"),
    }
    if fit["status"] != "ok":
        return nan_block
    predicted = predict_sigmoid_loglog(ordered["step"], float(fit["y_hi"]), float(fit["y_lo"]), float(fit["t0"]), float(fit["k"]))
    predicted_final = float(predicted[-1])
    predicted_min = float(predicted.min())
    return {
        "final_step": final_step,
        "actual_final": actual_final,
        "predicted_final": predicted_final,
        "abs_err_final": abs(predicted_final - actual_final),
        "rel_err_final": abs(predicted_final - actual_final) / actual_final,
        "actual_min": actual_min,
        "predicted_min": predicted_min,
        "abs_err_min": abs(predicted_min - actual_min),
        "rel_err_min": abs(predicted_min - actual_min) / actual_min,
    }


def annotate_sigmoid_catalog(catalog: pd.DataFrame, runs_by_id: RunArchiveMap, run_ids: Sequence[str]) -> pd.DataFrame:
    """Copy the power-law catalog and mark rows with too few train points for a 4-parameter S-curve."""
    rows: list[dict[str, object]] = []
    for experiment in catalog.itertuples(index=False):
        _, train_points, test_points, _ = scope_series(
            experiment.metric, experiment.scope, float(experiment.train_fraction), runs_by_id, run_ids
        )
        n_train = len(train_points)
        rows.append(
            {
                "train_fraction": experiment.train_fraction,
                "metric": experiment.metric,
                "kind": experiment.kind,
                "scope": experiment.scope,
                "train_set": experiment.train_set,
                "eval_set": experiment.eval_set,
                "n_train": n_train,
                "n_test": len(test_points),
                "sigmoid_eligible": n_train >= SIGMOID_MIN_TRAIN_POINTS,
            }
        )
    return pd.DataFrame(rows)


def run_sigmoid_sweep(catalog: pd.DataFrame, runs_by_id: RunArchiveMap, run_ids: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a log-log S-curve on every catalog row and score tails plus final/min loss.

    Returns:
        ``fits``: one row per (train_fraction, metric, scope) with plateaus,
        ``t0``, ``k``, and tail errors. ``endpoints``: one row per target run
        under that fit. Display cells consume both. Ineligible rows keep NaNs.
    """
    fit_rows: list[dict[str, object]] = []
    endpoint_rows: list[dict[str, object]] = []
    for experiment in catalog.itertuples(index=False):
        members, train_points, test_points, full_by_run = scope_series(
            experiment.metric, experiment.scope, float(experiment.train_fraction), runs_by_id, run_ids
        )
        fit = fit_sigmoid_loglog(train_points["step"], train_points["value"])
        train_errors = sigmoid_errors(train_points, fit)
        test_errors = sigmoid_errors(test_points, fit)
        fit_rows.append(
            {
                "train_fraction": experiment.train_fraction,
                "metric": experiment.metric,
                "kind": experiment.kind,
                "scope": experiment.scope,
                "train_set": experiment.train_set,
                "eval_set": experiment.eval_set,
                "n_train": int(fit["n"]),
                "n_test": len(test_points),
                "status": fit["status"],
                "y_hi": fit["y_hi"],
                "y_lo": fit["y_lo"],
                "t0": fit["t0"],
                "k": fit["k"],
                "train_mae": train_errors["mae"],
                "test_mae": test_errors["mae"],
                "train_log_mse": train_errors["log_mse"],
                "test_log_mse": test_errors["log_mse"],
            }
        )
        for target_run, full_points in full_by_run.items():
            endpoint_rows.append(
                {
                    "train_fraction": experiment.train_fraction,
                    "metric": experiment.metric,
                    "kind": experiment.kind,
                    "scope": experiment.scope,
                    "target_run": target_run,
                    "status": fit["status"],
                    **sigmoid_endpoint_accuracy(full_points, fit),
                }
            )
    return pd.DataFrame(fit_rows), pd.DataFrame(endpoint_rows)
