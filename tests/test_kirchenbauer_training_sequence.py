import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from ciphers.kirchenbauer_et_al.src.configuration_kl_fineweb import load_training_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "ciphers/kirchenbauer_et_al/scripts/run_bit_training_sequence.py"
WANDB_PROJECT = "stego-kirchenbauer-prefix-kl"


def _run_with_fake_conda(tmp_path: Path, *arguments: str, failing_configuration: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run the sequence without training and optionally fail at one configuration.

    Args:
        tmp_path: Pytest-owned directory used for the fake executable and its
            invocation log. No repository or artifact files are written here.
        arguments: Launcher CLI arguments; empty selects the default model and all bits.
        failing_configuration: Config filename whose simulated training process
            returns a failure. ``None`` makes all simulated processes succeed.

    Returns:
        The completed sequence process. Callers inspect its status and the
        newline-delimited ``calls.log`` file to verify orchestration behavior.
    """
    fake_binary_directory = tmp_path / "bin"
    fake_binary_directory.mkdir()
    fake_conda_path = fake_binary_directory / "conda"
    fake_conda_path.write_text(
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$TRAINING_SEQUENCE_CALL_LOG"
if [[ -n "${TRAINING_SEQUENCE_FAIL_CONFIG:-}" && "$*" == *"$TRAINING_SEQUENCE_FAIL_CONFIG"* ]]; then
  exit 23
fi
"""
    )
    fake_conda_path.chmod(0o755)

    environment = os.environ.copy()
    environment["PATH"] = f"{fake_binary_directory}:{environment['PATH']}"
    environment["TRAINING_SEQUENCE_CALL_LOG"] = str(tmp_path / "calls.log")
    if failing_configuration is not None:
        environment["TRAINING_SEQUENCE_FAIL_CONFIG"] = failing_configuration

    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *arguments],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _recorded_calls(tmp_path: Path) -> list[str]:
    return (tmp_path / "calls.log").read_text().splitlines()


@pytest.mark.parametrize("model_size", [None, "0.6b", "1.7b", "4b"])
def test_training_sequence_runs_largest_to_smallest_in_one_wandb_project(tmp_path: Path, model_size: str | None) -> None:
    """Cover all successful jobs and ordering; model, data, Conda, and W&B execution are omitted."""
    result = _run_with_fake_conda(tmp_path, *(["--model-size", model_size] if model_size else []))

    assert result.returncode == 0
    calls = _recorded_calls(tmp_path)
    assert [Path(call.split("--config ", 1)[1].split(" ", 1)[0]).name for call in calls] == [
        "eight_bit_training_run.yaml",
        "four_bit_training_run.yaml",
        "two_bit_training_run.yaml",
        "one_bit_training_run.yaml",
    ]
    assert all(f"--wandb-project {WANDB_PROJECT}" in call for call in calls)
    for call, bit_count in zip(calls, (8, 4, 2, 1), strict=True):
        command = shlex.split(call)
        assert command[:7] == ["run", "--no-capture-output", "-n", "stego", "python", "-m", "ciphers.kirchenbauer_et_al.src.train_kl_fineweb"]
        config = load_training_config(command[command.index("--config") + 1])
        assert config.model == f"Qwen/Qwen3-{(model_size or '0.6b').upper()}-Base"
        assert config.n_bits == bit_count
    assert "[1/4] Starting" in result.stdout
    assert "[4/4] Completed" in result.stdout
    assert "Command: conda run" in result.stdout
    assert "Finished all 4 training job(s)." in result.stdout


def test_training_sequence_stops_after_a_failed_job(tmp_path: Path) -> None:
    """Cover a middle-job failure partition; failures before or after process launch are omitted."""
    result = _run_with_fake_conda(tmp_path, failing_configuration="four_bit_training_run.yaml")

    assert result.returncode == 23
    calls = _recorded_calls(tmp_path)
    assert len(calls) == 2
    assert "eight_bit_training_run.yaml" in calls[0]
    assert "four_bit_training_run.yaml" in calls[1]

    assert "Failed:" in result.stderr
    assert "exit status 23" in result.stderr
    assert "Finished all" not in result.stdout


@pytest.mark.parametrize("model_size", ["0.6b", "1.7b", "4b"])
@pytest.mark.parametrize("bits", ["1", "2", "4", "8"])
def test_single_experiment_selection(tmp_path: Path, model_size: str, bits: str) -> None:
    """Cover the 3 sizes x 4 bit counts through CLI and YAML loading; training execution is omitted."""
    result = _run_with_fake_conda(tmp_path, "--model-size", model_size, "--bits", bits)

    assert result.returncode == 0
    calls = _recorded_calls(tmp_path)
    assert len(calls) == 1
    command = shlex.split(calls[0])
    config = load_training_config(command[command.index("--config") + 1])
    assert config.model == f"Qwen/Qwen3-{model_size.upper()}-Base"
    assert config.n_bits == int(bits)
    assert f"Completed {model_size}, {bits} bit(s)." in result.stdout


@pytest.mark.parametrize("arguments", [("--model-size", "7b"), ("--bits", "3"), ("--bits",), ("--model-size",), ("--unknown",)])
def test_invalid_selection_never_launches_training(tmp_path: Path, arguments: tuple[str, ...]) -> None:
    """Cover unsupported choices, missing values, and unknown flags; OS launch failures are omitted."""
    result = _run_with_fake_conda(tmp_path, *arguments)

    assert result.returncode == 2
    assert "Error:" in result.stderr
    assert not (tmp_path / "calls.log").exists()


def test_help_does_not_launch_training(tmp_path: Path) -> None:
    """Cover the informational CLI path; terminal formatting variations are omitted."""
    result = _run_with_fake_conda(tmp_path, "--help")

    assert result.returncode == 0
    assert "--model-size" in result.stdout
    assert "--bits" in result.stdout
    assert not (tmp_path / "calls.log").exists()
