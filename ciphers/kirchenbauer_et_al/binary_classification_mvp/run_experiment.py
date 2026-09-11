#!/usr/bin/env python3
"""Run the experiment's production stages from one JSON/YAML configuration."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared import (
    ARTIFACTS_DIR_ENV,
    EXPERIMENT_DIR,
    REPO_ROOT,
    artifact_paths,
    load_experiment_config,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = artifact_paths()
    print(f"Artifact root: {paths.root}", flush=True)
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
        print(f"Running {stage}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True, cwd=REPO_ROOT, env=child_environment)


if __name__ == "__main__":
    main()
