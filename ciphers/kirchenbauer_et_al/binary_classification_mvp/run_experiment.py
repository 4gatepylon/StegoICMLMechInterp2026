#!/usr/bin/env python3
"""Run the experiment's production stages from one JSON/YAML configuration."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from shared import EXPERIMENT_DIR, artifact_paths, load_experiment_config

STAGE_SCRIPTS = {
    "prefix": EXPERIMENT_DIR / "finetune_prefix.py",
    "encoding": EXPERIMENT_DIR / "finetune_encoding.py",
    "evaluation": EXPERIMENT_DIR / "evaluate_generation.py",
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
        choices=tuple(STAGE_SCRIPTS),
        default=tuple(STAGE_SCRIPTS),
        help="Stages to run in order; defaults to the complete pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = artifact_paths()
    print(f"Artifact root: {paths.root}", flush=True)
    config_path = Path(args.config).expanduser().resolve()
    # Validate the complete configuration before starting an expensive stage.
    load_experiment_config(str(config_path))
    for stage in args.stages:
        command = [
            sys.executable,
            str(STAGE_SCRIPTS[stage]),
            "--config",
            str(config_path),
        ]
        print(f"Running {stage}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
