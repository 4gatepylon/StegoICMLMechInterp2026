#!/usr/bin/env python3
"""Run the experiment's production stages from one JSON/YAML configuration."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.artifacts import artifact_paths
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.configuration import load_experiment_config
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.constants import (
    ARTIFACTS_DIR_ENV,
    EXPERIMENT_DIR,
    REPO_ROOT,
)

STAGE_MODULES = {
    "prefix": "ciphers.kirchenbauer_et_al.binary_classification_mvp.finetune_prefix",
    "encoding": "ciphers.kirchenbauer_et_al.binary_classification_mvp.finetune_encoding",
    "evaluation": "ciphers.kirchenbauer_et_al.binary_classification_mvp.evaluate_generation",
}
DEFAULT_CONFIG = EXPERIMENT_DIR / "configurations" / "official.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="JSON or YAML experiment configuration (relative to the current directory).",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=tuple(STAGE_MODULES),
        default=tuple(STAGE_MODULES),
        help="Stages to run in order; defaults to the complete pipeline.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only select and log corpus data; do not load or train models.",
    )
    parser.add_argument(
        "--cache-dir",
        help="Override the shared cache directory (defaults to ARTIFACTS_DIR/cache).",
    )
    parser.add_argument("--batch-size", type=int, help="Override the training batch size.")
    parser.add_argument("--eval-batch-size", type=int, help="Override the teacher-forced evaluation batch size.")
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "online", "offline"),
        help="Override the configuration W&B mode for both training stages.",
    )
    parser.add_argument("--wandb-project", help="Override the W&B project for both training stages.")
    parser.add_argument("--wandb-run-name", help="Override the base W&B run name for both training stages.")
    parser.add_argument("--wandb-entity", help="Override the W&B entity for both training stages.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = artifact_paths()
    print(f"Artifact root: {paths.root}", flush=True)
    cache_dir = str(Path(args.cache_dir).expanduser().resolve()) if args.cache_dir is not None else None
    config_path = Path(args.config).expanduser().resolve()
    # Validate the complete configuration before starting an expensive stage.
    load_experiment_config(str(config_path))
    stages = ("prefix",) if args.dry_run else args.stages
    child_environment = os.environ.copy()
    child_environment[ARTIFACTS_DIR_ENV] = str(paths.root)
    optimization_flags = ["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []
    for stage in stages:
        command = [
            sys.executable,
            *optimization_flags,
            "-m",
            STAGE_MODULES[stage],
            "--config",
            str(config_path),
        ]
        if args.dry_run:
            command.append("--dry-run")
        elif stage in {"prefix", "encoding"}:
            for flag, value in (
                ("--batch-size", args.batch_size),
                ("--eval-batch-size", args.eval_batch_size),
                ("--wandb-mode", args.wandb_mode),
                ("--wandb-project", args.wandb_project),
                ("--wandb-run-name", args.wandb_run_name),
                ("--wandb-entity", args.wandb_entity),
            ):
                if value is not None:
                    command.extend((flag, str(value)))
        if cache_dir is not None:
            command.extend(("--cache-dir", cache_dir))
        print(f"Running {stage}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True, cwd=REPO_ROOT, env=child_environment)


if __name__ == "__main__":
    main()
