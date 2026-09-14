"""Exercise generated presets and exact directory verification without training."""

import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import pytest
from pydantic_yaml import parse_yaml_raw_as

from ciphers.kirchenbauer_et_al.scripts import generate_experiment_configs
from ciphers.kirchenbauer_et_al.src.configuration_kl_fineweb import PrefixKLTrainingConfig


def _run_cli(arguments: list[str]) -> tuple[int, str]:
    """Capture a CLI invocation without launching a subprocess.

    Args:
        arguments: Generator CLI arguments without the program name.

    Returns:
        ``(exit_status, output)`` where the status is zero on normal return or
        the integer ``SystemExit`` code, and output combines stdout and stderr.
        Tests use this to inspect success and failure reports under an isolated
        repository root. Unexpected exceptions propagate and fail the test.
    """
    output = StringIO()
    with redirect_stdout(output), redirect_stderr(output):
        try:
            generate_experiment_configs.main(arguments)
        except SystemExit as error:
            return int(error.code), output.getvalue()
    return 0, output.getvalue()


def _snapshot(directory: Path) -> dict[Path, tuple[bytes, int]]:
    """Capture file contents and modification times to detect accidental writes.

    Args:
        directory: Existing output directory to inspect recursively.

    Returns:
        A mapping from relative file paths to ``(contents, mtime_ns)`` tuples.
        Contents are raw bytes and modification times are integer nanoseconds.
        Callers compare snapshots before and after read-only checks.
    """
    return {path.relative_to(directory): (path.read_bytes(), path.stat().st_mtime_ns) for path in directory.rglob("*") if path.is_file()}


@pytest.fixture
def generated_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Generate presets under an isolated repository root for CLI behavior tests."""
    monkeypatch.setattr(generate_experiment_configs, "REPO_ROOT", tmp_path)
    status, output = _run_cli(["generate", "--output-dir", "experiments"])
    assert status == 0, output
    return tmp_path / "experiments"


def test_generation_produces_loadable_model_bit_matrix(generated_directory: Path) -> None:
    """Cover all 3 sizes x 4 bit counts and output isolation; model execution is omitted."""
    configs = [parse_yaml_raw_as(PrefixKLTrainingConfig, path.read_text()) for path in generated_directory.rglob("*.yaml")]

    assert len(configs) == 12
    assert {(config.model, config.n_bits) for config in configs} == {(f"Qwen/Qwen3-{size}-Base", bits) for size in ("0.6B", "1.7B", "4B") for bits in (1, 2, 4, 8)}
    assert len({config.run_name for config in configs}) == 12
    shared_settings = [config.model_dump(exclude={"model", "n_bits", "run_name"}) for config in configs]
    assert all(settings == shared_settings[0] for settings in shared_settings)


def test_second_generation_is_identical_in_new_and_existing_directories(generated_directory: Path) -> None:
    """Cover deterministic fresh output and overwriting the same destination; I/O faults are omitted."""
    original_bytes = {path: contents for path, (contents, _) in _snapshot(generated_directory).items()}
    for output_dir in ("second-output", "experiments"):
        status, output = _run_cli(["generate", "--output-dir", output_dir])
        assert status == 0, output
        actual = _snapshot(generated_directory.parent / output_dir)
        assert {path: contents for path, (contents, _) in actual.items()} == original_bytes


def test_matching_check_does_not_write(generated_directory: Path) -> None:
    """Cover successful exact comparison and read-only behavior; concurrent modification is omitted."""
    before = _snapshot(generated_directory)
    status, output = _run_cli(["check", "--output-dir", "experiments"])

    assert status == 0, output
    assert "exact match" in output
    assert _snapshot(generated_directory) == before


@pytest.mark.parametrize("mutation", ["value", "whitespace", "line_endings", "missing", "extra_yaml", "extra_other"])
def test_check_reports_mismatches_without_modifying_files(generated_directory: Path, mutation: str) -> None:
    """Cover value/byte changes, missing files, and extra YAML/non-YAML files; permissions are omitted."""
    changed_file = generated_directory / "qwen3-0.6b/two_bit_training_run.yaml"
    expected_status = "Changed:"
    if mutation == "value":
        changed_file.write_bytes(changed_file.read_bytes().replace(b"n_bits: 2", b"n_bits: 3"))
    elif mutation == "whitespace":
        changed_file.write_bytes(changed_file.read_bytes() + b"\n")
    elif mutation == "line_endings":
        changed_file.write_bytes(changed_file.read_bytes().replace(b"\n", b"\r\n"))
    elif mutation == "missing":
        changed_file.unlink()
        expected_status = "Missing:"
    else:
        changed_file = generated_directory / ("extra.yaml" if mutation == "extra_yaml" else "extra.txt")
        changed_file.write_text("unexpected\n")
        expected_status = "Unexpected:"
    before = _snapshot(generated_directory)

    status, output = _run_cli(["check", "--output-dir", "experiments"])

    assert status == 1
    assert f"{expected_status} {changed_file.relative_to(generated_directory)}" in output
    assert _snapshot(generated_directory) == before


def test_check_missing_directory_does_not_create_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover an absent output tree; partially missing trees are covered separately."""
    monkeypatch.setattr(generate_experiment_configs, "REPO_ROOT", tmp_path)
    status, output = _run_cli(["check", "--output-dir", "absent"])

    assert status == 1
    assert output.count("Missing:") == 12
    assert not (tmp_path / "absent").exists()


def test_generation_repairs_presets_and_preserves_unrelated_files(generated_directory: Path) -> None:
    """Cover replacing edited presets while retaining unrelated content; concurrent writers are omitted."""
    (generated_directory / "qwen3-0.6b/two_bit_training_run.yaml").write_text("edited\n")
    unrelated = generated_directory / "notes.txt"
    unrelated.write_text("keep me\n")
    status, output = _run_cli(["generate", "--output-dir", "experiments"])

    assert status == 0, output
    assert unrelated.read_text() == "keep me\n"
    status, output = _run_cli(["check", "--output-dir", "experiments"])
    assert status == 1
    assert "Unexpected: notes.txt" in output
    assert "Changed:" not in output


@pytest.mark.parametrize("command", ["generate", "check"])
@pytest.mark.parametrize("output_dir", ["../outside", "/absolute"])
def test_output_path_must_stay_relative_to_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, output_dir: str) -> None:
    """Cover traversal and absolute-path rejection for both commands; symlink races are omitted."""
    monkeypatch.setattr(generate_experiment_configs, "REPO_ROOT", tmp_path)
    status, output = _run_cli([command, "--output-dir", output_dir])

    assert status == 2
    assert "inside the repository root" in output
    assert not list(tmp_path.iterdir())


def test_checked_in_presets_match_generator() -> None:
    """Cover checked-in output against fresh generation; this catches stale YAML after template or sweep edits."""
    status, output = _run_cli(["check"])

    assert status == 0, output


def test_both_commands_run_without_third_party_packages(tmp_path: Path) -> None:
    """Cover direct script execution with site packages and Python environment hooks disabled; training is omitted."""
    script_path = tmp_path / "ciphers/kirchenbauer_et_al/scripts/generate_experiment_configs.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_bytes(Path(generate_experiment_configs.__file__).read_bytes())
    for command in ("generate", "check"):
        result = subprocess.run(
            [sys.executable, "-I", "-S", str(script_path), command, "--output-dir", "experiments"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
    assert len(list((tmp_path / "experiments").rglob("*.yaml"))) == 12
