"""Launch model-size and bit-count presets sequentially in the stego environment."""

import shlex
import subprocess
from pathlib import Path

import click

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGURATION_DIRECTORY = Path("ciphers/kirchenbauer_et_al/experiments")
BIT_NAMES = {8: "eight", 4: "four", 2: "two", 1: "one"}


@click.command(help="Train one model size in the stego environment. Prints each command and stops on failure.")
@click.option("--model-size", type=click.Choice(["0.6b", "1.7b", "4b"]), default="0.6b", show_default=True)
@click.option("--bits", type=click.Choice(["1", "2", "4", "8"]), help="Run only this bit count; omitted runs 8, 4, 2, 1 in order.")
def main(model_size: str, bits: str | None) -> None:
    """Train the selected presets and stream each job's output to the terminal.

    Args:
        model_size: Qwen3 base-model size, validated by Click as ``0.6b``,
            ``1.7b``, or ``4b``. Selects the matching experiment directory.
        bits: A single bit count, validated by Click as ``1``, ``2``, ``4``, or
            ``8``. ``None`` selects all four counts in descending order.

    Returns:
        ``None`` after every selected job succeeds. Each subprocess runs from
        the repository root through Conda's ``stego`` environment, consumes a
        repository-relative YAML, and inherits the process environment and
        standard streams. The training entry point requires a completed FineWeb
        cache under ``STEGO_ARTIFACTS_DIR`` and writes checkpoints there using
        the YAML's run name. W&B runs use ``stego-kirchenbauer-prefix-kl``.
        A failed training job terminates the launcher with its exit status
        (or ``128 + signal`` for signal termination); later jobs are not started.
    """
    selected_bits = [int(bits)] if bits is not None else list(BIT_NAMES)
    click.echo(f"Training Qwen3-{model_size.upper()}-Base: bit counts {', '.join(map(str, selected_bits))}")
    click.echo(f"Working directory: {REPO_ROOT}")
    for job_index, bit_count in enumerate(selected_bits, start=1):
        config_path = CONFIGURATION_DIRECTORY / f"qwen3-{model_size}" / f"{BIT_NAMES[bit_count]}_bit_training_run.yaml"
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "stego",
            "python",
            "-m",
            "ciphers.kirchenbauer_et_al.src.train_kl_fineweb",
            "--config",
            str(config_path),
            "--wandb-project",
            "stego-kirchenbauer-prefix-kl",
        ]
        click.echo(f"[{job_index}/{len(selected_bits)}] Starting {model_size}, {bit_count} bit(s); config: {config_path}")
        click.echo(f"Command: {shlex.join(command)}")
        try:
            result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        except OSError as error:
            raise click.ClickException(f"Could not start {config_path}: {error}") from error
        if result.returncode:
            click.echo(f"Failed: {config_path} (exit status {result.returncode}). Stopping sequence.", err=True)
            raise SystemExit(result.returncode if result.returncode > 0 else 128 - result.returncode)
        click.echo(f"[{job_index}/{len(selected_bits)}] Completed {model_size}, {bit_count} bit(s).")
    click.echo(f"Finished all {len(selected_bits)} training job(s).")


if __name__ == "__main__":
    main()
