"""Compare bit counts and objective settings at a fixed Qwen3-0.6B token budget.

Hypothesis: some settings improve held-out prefix/data losses; flat or diverging
losses suggest optimization or capacity limits. This does not test bit recovery
or text quality, and total losses are not comparable across objective weights.

Activate stego, prepare fineweb-500k, set STEGO_ARTIFACTS_DIR, and authenticate
W&B as described in README.md. Run this file with --dry-run to preview, or
without it to train sequentially on one visible GPU. Edit the grids below.
Fixed train.py settings: LoRA rank=32, alpha=16, dropout=0.05; data length=1024,
validation samples=256, warmup=50 steps, eval/save/log every 4/32/1 steps,
bfloat16, block encoding, padding rejected, student BOS off, teacher BOS on,
and fineweb-500k cache. Document-length upper bounds are unlimited.
The grid's token budget and batches give 1024 steps with accumulation=16.
Each run retains 32 PEFT checkpoints (steps 32, 64, ..., 1024): adapter weights
and tokenizer/Trainer metadata, without full model weights or optimizer state.
Outputs go under $STEGO_ARTIFACTS_DIR/<run-name>/. Use a fresh artifacts root
for independent repeats: identical settings reuse the same output directory.
"""

from itertools import product
from pathlib import Path
import shlex
import subprocess
import sys

import click


# Keys are train.py CLI names, except the paired loss/alpha entry.
# Singleton lists hold settings fixed; None omits alpha when it has no effect.
GRID = {
    "model": ["Qwen/Qwen3-0.6B-Base"],
    "n-bits": [1, 2, 4],
    "lr": [1e-4, 3e-4, 1e-3],
    "loss-type/alpha": [("nll", 0.1), ("nll", 1.0), ("ignore_prefix", None)],
    "delta": [2.0, 4.0],
    "global-batch-size": [32],
    "local-batch-size": [2],
    "num-training-tokens": [33_554_432],  # 1024 steps * batch 32 * 1024 data tokens; excludes prefixes/validation.
    "min-gpt2-document-tokens": [756],
    "min-qwen-document-tokens": [1024],
}
TRAIN_MODULE = "ciphers.kirchenbauer_et_al.experiments.E20260916_qwen3_0_6b_peft_convergence.train"
REPO_ROOT = Path(__file__).resolve().parents[4]


@click.command()
@click.option("--dry-run", is_flag=True, help="Print all training commands without launching them.")
def main(dry_run: bool) -> None:
    """Print the grid and, unless dry_run, launch each run in a fresh process.

    Returns None. Each child uses the active Python environment and repo root;
    train.py validates its settings and records outputs. The first failed run
    stops the sweep. Alpha is swept only for nll because ignore_prefix ignores it.
    """
    combinations = [dict(zip(GRID, values)) for values in product(*GRID.values())]
    for index, params in enumerate(combinations, 1):
        command = [sys.executable, "-m", TRAIN_MODULE]
        for name, value in params.items():
            if name == "loss-type/alpha":
                loss_type, alpha = value
                command.extend(["--loss-type", loss_type])
                if alpha is not None:
                    command.extend(["--alpha", str(alpha)])
            else:
                command.extend([f"--{name}", str(value)])
        click.echo(f"[{index}/{len(combinations)}] {shlex.join(command)}")
        if not dry_run:
            try:
                subprocess.run(command, cwd=REPO_ROOT, check=True)
            except subprocess.CalledProcessError as error:
                raise click.ClickException(f"Run {index} failed (exit {error.returncode}); sweep stopped.") from error


if __name__ == "__main__":
    main()
