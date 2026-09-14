"""Planning tests cover coordinate resolution, archive arithmetic and file boundaries.

No models, dataset contents, GPUs, W&B network access, convergence, or real
wall-clock performance are tested. The shipped spec is also generated and
reviewed manually as an integration artifact.
"""

import json
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic_yaml import parse_yaml_raw_as

from ciphers.kirchenbauer_et_al.src import sweep_kl_fineweb as sweep
from ciphers.kirchenbauer_et_al.src.configuration_kl_fineweb import build_sft_config, parse_args

SPEC = "ciphers/kirchenbauer_et_al/experiments/hyperparameter_sweep.yaml"


def test_grid_preserves_document_budget_and_resolves_all_model_batches() -> None:
    """Cover every model/batch/objective partition, including local-batch reduction."""
    spec = parse_yaml_raw_as(sweep.SweepSpec, (sweep.REPO_ROOT / SPEC).read_text())
    spec.selected_models = [preset.name for preset in spec.models]
    manifest = sweep.generate_manifest(spec, "sweeps/test", "test-digest")
    expected = {
        (preset.model, bits, lr, batch, alpha, strategy)
        for preset in spec.models
        for bits in spec.bits
        for lr in spec.learning_rates
        for batch in spec.global_batch_sizes
        for alpha in spec.alphas
        for strategy in spec.strategies
    }
    actual = set()
    for trial in manifest.trials:
        config = trial.training
        actual.add((config.model, config.n_bits, config.learning_rate, config.global_batch_size, None if config.loss_mode == "ignore_prefix" else config.alpha, config.strategy))
        assert config.max_steps * config.global_batch_size == spec.training_documents
        assert config.warmup_steps * config.global_batch_size == spec.warmup_documents
        assert config.eval_steps * config.global_batch_size == spec.evaluation_interval_documents
        assert config.per_device_batch_size * trial.world_size * trial.resolved_gradient_accumulation_steps == config.global_batch_size
        assert config.per_device_batch_size <= next(p.local_batch_size for p in spec.models if p.name == trial.model_preset)
    assert actual == expected
    assert len(manifest.trials) == len(expected)


def test_manifest_rejects_budget_edit_and_spec_rejects_incompatible_batch() -> None:
    """Cover inconsistent derived fields and a non-divisible batch, not Pydantic internals."""
    spec = parse_yaml_raw_as(sweep.SweepSpec, (sweep.REPO_ROOT / SPEC).read_text())
    manifest = sweep.generate_manifest(spec, "sweeps/test", "digest").model_dump()
    manifest["trials"][0]["training"]["max_steps"] += 1
    with pytest.raises(ValueError, match="training_documents"):
        sweep.Manifest.model_validate(manifest)
    invalid = spec.model_dump()
    invalid["global_batch_sizes"] = [12]
    with pytest.raises(ValueError, match="divisible"):
        sweep.SweepSpec.model_validate(invalid)


def test_calibration_separates_eval_and_ignores_visible_gpu_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover interleaved eval rows and misleading GPU metadata using exact elapsed times."""
    with zipfile.ZipFile(tmp_path / "run.zip", "w") as archive:
        archive.writestr(
            "run.json",
            json.dumps({"url": "https://wandb.ai/example/project/runs/test", "config": {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 4, "max_length": 4096}}),
        )
        archive.writestr("files/wandb-metadata.json", json.dumps({"gpu_count": 8, "args": ["--global-batch-size", "32", "--validation-samples", "4"]}))
        archive.writestr(
            "history.jsonl",
            "\n".join(
                json.dumps(row)
                for row in [
                    {"_runtime": 100, "train/global_step": 1, "train/loss": 2},
                    {"_runtime": 108, "train/global_step": 1, "eval/runtime": 8},
                    {"_runtime": 118, "train/global_step": 2, "train/loss": 1.5},
                    {"_runtime": 130, "train/global_step": 3, "train/loss": 1},
                ]
            ),
        )
    monkeypatch.setattr(sweep, "REPO_ROOT", tmp_path)
    result = sweep.calibrate_archive(sweep.CalibrationSource(archive="run.zip", global_batch_size=32, validation_documents=4))
    assert result.inferred_world_size == 4
    assert result.median_step_seconds == 11
    assert result.p95_step_seconds == 12
    assert result.device_seconds_per_document == 1.5
    assert result.evaluation_seconds_per_document == 2


def test_estimator_accounts_for_documents_evaluations_steps_and_margin() -> None:
    """Cover all global batches with a synthetic calibration; no empirical accuracy claim."""
    spec = parse_yaml_raw_as(sweep.SweepSpec, (sweep.REPO_ROOT / SPEC).read_text())
    trials = sweep.generate_manifest(spec, "sweeps/test", "digest").trials
    calibration = sweep.Calibration(
        archive="test.zip",
        sha256="digest",
        source_url="test",
        global_batch_size=32,
        inferred_world_size=4,
        max_length=4096,
        intervals=2,
        median_step_seconds=11,
        p95_step_seconds=12,
        device_seconds_per_document=1.5,
        evaluation_seconds_per_document=2,
    )
    for trial in trials:
        result = sweep.estimate_trial(trial, spec.timing, [calibration])
        assert result.training_seconds == pytest.approx(1536 + trial.training.max_steps * 0.1)
        assert result.evaluation_seconds == 128
        assert result.total_seconds == pytest.approx((result.training_seconds + 128 + 180) * 1.2)


def test_cli_roundtrip_refuses_overwrite_and_escaping_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover manifest write/read, existing-file protection and path traversal rejection."""
    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    runner = CliRunner()
    args = ["generate", "--spec", SPEC, "--output", "sweeps/test/manifest.json", "--model", "qwen3-0.6b"]
    result = runner.invoke(sweep.cli, args)
    assert result.exit_code == 0, result.output
    manifest_path = tmp_path / "sweeps/test/manifest.json"
    manifest = sweep.Manifest.model_validate_json(manifest_path.read_text())
    assert {trial.model_preset for trial in manifest.trials} == {"qwen3-0.6b"}
    assert runner.invoke(sweep.cli, ["generate", "--spec", SPEC, "--output", "manifest.json"]).exit_code == 0
    original = manifest_path.read_bytes()
    assert runner.invoke(sweep.cli, args).exit_code != 0
    assert manifest_path.read_bytes() == original
    assert runner.invoke(sweep.cli, ["generate", "--spec", SPEC, "--output", "../escape.json"]).exit_code != 0


def test_training_cli_accepts_gradient_clipping_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover clipping override propagation through the CLI and SFTConfig; omit model execution."""
    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    config = parse_args(["--config", "ciphers/kirchenbauer_et_al/experiments/one_bit_training_run.yaml", "--max-grad-norm", "2.5"])
    assert config.max_grad_norm == 2.5
    assert build_sft_config(config, grad_accumulation_steps=8).max_grad_norm == 2.5
