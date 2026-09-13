import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "ciphers/kirchenbauer_et_al/scripts/run_bit_training_sequence.sh"
WANDB_PROJECT = "stego-kirchenbauer-prefix-kl"


def _run_with_fake_conda(tmp_path: Path, *, failing_configuration: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run the sequence without training and optionally fail at one configuration.

    Args:
        tmp_path: Pytest-owned directory used for the fake executable and its
            invocation log. No repository or artifact files are written here.
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
        ["bash", str(SCRIPT_PATH)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _recorded_calls(tmp_path: Path) -> list[str]:
    return (tmp_path / "calls.log").read_text().splitlines()


def test_training_sequence_runs_largest_to_smallest_in_one_wandb_project(tmp_path: Path) -> None:
    """Cover all successful jobs and ordering; model, data, Conda, and W&B execution are omitted."""
    result = _run_with_fake_conda(tmp_path)

    assert result.returncode == 0
    calls = _recorded_calls(tmp_path)
    assert [Path(call.split("--config ", 1)[1].split(" ", 1)[0]).name for call in calls] == [
        "eight_bit_training_run.yaml",
        "four_bit_training_run.yaml",
        "two_bit_training_run.yaml",
        "one_bit_training_run.yaml",
    ]
    assert all(f"--wandb-project {WANDB_PROJECT}" in call for call in calls)


def test_training_sequence_stops_after_a_failed_job(tmp_path: Path) -> None:
    """Cover a middle-job failure partition; failures before or after process launch are omitted."""
    result = _run_with_fake_conda(tmp_path, failing_configuration="four_bit_training_run.yaml")

    assert result.returncode == 23
    calls = _recorded_calls(tmp_path)
    assert len(calls) == 2
    assert "eight_bit_training_run.yaml" in calls[0]
    assert "four_bit_training_run.yaml" in calls[1]
