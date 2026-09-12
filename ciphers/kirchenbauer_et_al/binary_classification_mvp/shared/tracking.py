"""Optional Weights & Biases tracking for training and validation metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


def init_wandb(
    *,
    mode: Literal["online", "offline", "disabled"],
    project: str,
    run_name: str,
    entity: str | None,
    stage: str,
    output_dir: Path,
    config: dict[str, Any],
) -> Run | None:
    """Start the stage-specific W&B run, or return None when tracking is disabled."""

    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("W&B tracking is enabled but wandb is not installed; install the experiment requirements") from error
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_run_name = f"{run_name}-{stage}"
    run = wandb.init(
        project=project,
        entity=entity,
        name=stage_run_name,
        group=run_name,
        job_type=stage,
        mode=mode,
        dir=str(output_dir),
        config=config,
    )
    run.define_metric("trainer/step")
    run.define_metric("*", step_metric="trainer/step")
    print(f"W&B tracking: project={project!r}, run={stage_run_name!r}, mode={mode!r}", flush=True)
    return run


def init_wandb_from_args(args: argparse.Namespace, *, stage: str, output_dir: Path) -> Run | None:
    """Start W&B from the resolved experiment configuration and CLI arguments."""

    config = args.experiment_config.model_dump(mode="json")
    config.update(
        stage=stage,
        wandb={
            "mode": args.wandb_mode,
            "project": args.wandb_project,
            "run_name": args.wandb_run_name,
            "entity": args.wandb_entity,
        },
    )
    return init_wandb(
        mode=args.wandb_mode,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        entity=args.wandb_entity,
        stage=stage,
        output_dir=output_dir,
        config=config,
    )


def log_metric_records(run: Run | None, records: Sequence[dict[str, Any]]) -> None:
    """Log separate loss and color-mass curves for each split and signal."""

    if run is None or not records:
        return
    step = int(records[0]["step"])
    payload: dict[str, int | float] = {"trainer/step": step}
    for record in records:
        split = str(record["split"])
        signal = str(record["signal"])
        payload[f"loss/{split}/{signal}"] = float(record["kl"])
        for metric in (
            "expected_red",
            "expected_green",
            "expected_uncolored",
            "red_rate",
            "green_rate",
        ):
            payload[f"policy/{split}/{signal}/{metric}"] = float(record[metric])
    run.log(payload)


def finish_wandb(run: Run | None) -> None:
    """Finish a W&B run without affecting tracking-disabled executions."""

    if run is not None:
        run.finish()
