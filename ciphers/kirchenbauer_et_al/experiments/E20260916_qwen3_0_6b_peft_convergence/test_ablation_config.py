"""Exercise sweep commands, invalid inputs, naming, and trainer wiring on CPU.

Partitions include all three models/three bit lengths, both loss types, zero and
positive objective weights, invalid models, and nonpositive/nonfinite numbers.
Document filtering covers CLI forwarding of nested bounds, invalid ranges,
the selected tokenizer, and distinct filtered-run output names.
Trainer, tokenizer, dataset, and SFT construction are mocked: GPU training,
model downloads, distributed execution, and live W&B delivery are omitted.
"""

import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import get_args
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from ciphers.kirchenbauer_et_al.experiments.E20260916_qwen3_0_6b_peft_convergence import run_sweep, train
from ciphers.kirchenbauer_et_al.src.configuration_kl_fineweb import configure_wandb_environment, gradient_accumulation_steps

README = Path(train.__file__).with_name("README.md")
SWEEP_COMMANDS = [shlex.split(line)[3:] for line in README.read_text().splitlines() if line.startswith("python -m ") and "--model " in line]


@pytest.mark.parametrize("arguments", SWEEP_COMMANDS)
def test_documented_sweep_commands(arguments: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute each README command through Click; verify budget, batching and identity."""
    build = Mock()
    monkeypatch.setattr(train, "build_trainer", build)
    result = CliRunner().invoke(train.main, arguments)
    assert result.exit_code == 0, result.output
    config = build.call_args.args[0]
    assert config.model == arguments[arguments.index("--model") + 1]
    assert config.n_bits == int(arguments[arguments.index("--n-bits") + 1])
    assert config.max_steps * config.data_length * config.global_batch_size == config.num_training_tokens
    assert gradient_accumulation_steps(config, world_size=1) == 16
    assert gradient_accumulation_steps(config, world_size=4) == 4
    assert f"{config.n_bits}bit-lr0.0003-gb{config.global_batch_size}" in config.run_name
    build.return_value.train.assert_called_once_with()


def test_sweep_covers_full_cartesian_product() -> None:
    """Catch missing/duplicated commands in the documented model × bits sweep."""
    combinations = [(args[args.index("--model") + 1], int(args[args.index("--n-bits") + 1])) for args in SWEEP_COMMANDS]
    assert len(combinations) == len(set(combinations))
    assert set(combinations) == {(model, bits) for model in get_args(train.QwenModel) for bits in (1, 2, 4)}
    run_names = [train.experiment_config(local_batch_size=2, global_batch_size=32, model=model, n_bits=bits).run_name for model, bits in combinations]
    assert len(set(run_names)) == len(combinations)


@pytest.mark.parametrize("loss_type", ["nll", "ignore_prefix"])
@pytest.mark.parametrize("prepend_student_bos", [True, False])
@pytest.mark.parametrize("reject_padding", [True, False])
def test_objective_overrides_reach_trainer(loss_type: str, reject_padding: bool, prepend_student_bos: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Forward nondefault knobs through Click/config/trainer and collator/validation."""
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("WANDB_PROJECT", "old-project")
    monkeypatch.setenv("WANDB_TAGS", "")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    sft_args = SimpleNamespace(output_dir=str(tmp_path))
    monkeypatch.setattr(train, "build_sft_config", Mock(return_value=sft_args))
    dataset = Mock()
    monkeypatch.setattr(train, "load_fineweb_cache", Mock(return_value=dataset))
    monkeypatch.setattr(train.AutoTokenizer, "from_pretrained", Mock())
    trainer = Mock()
    monkeypatch.setattr(train, "PrefixKLTrainer", trainer)
    result = CliRunner().invoke(
        train.main,
        [
            "--reject-document-padding" if reject_padding else "--allow-document-padding",
            "--prepend-student-bos" if prepend_student_bos else "--no-prepend-student-bos",
            "--lr",
            "0.001",
            "--model-name",
            "Qwen/Qwen3-4B-Base",
            "--n-bits",
            "4",
            "--loss-type",
            loss_type,
            "--alpha",
            "0.5",
            "--delta",
            "4",
            "--min-gpt2-document-tokens",
            "100",
            "--max-gpt2-document-tokens",
            "2000",
            "--min-qwen-document-tokens",
            "512",
            "--max-qwen-document-tokens",
            "2048",
        ],
    )
    assert result.exit_code == 0, result.output
    config = train.build_sft_config.call_args.args[0]
    assert config.learning_rate == 0.001
    assert config.run_name == f"qwen3-4b-4bit-lr0.001-gb128-{loss_type}-a0.5-d4-tokens{config.num_training_tokens}-gpt2-100-2000-qwen-512-2048"
    train.load_fineweb_cache.assert_called_once_with(
        config.dataset_cache_name,
        minimum_documents=config.validation_samples + config.max_steps * config.global_batch_size,
        min_gpt2_document_tokens=100,
        max_gpt2_document_tokens=2000,
        min_qwen_document_tokens=512,
        max_qwen_document_tokens=2048,
        tokenizer=train.AutoTokenizer.from_pretrained.return_value,
    )
    kwargs = trainer.call_args.kwargs
    assert kwargs["reject_document_padding"] is reject_padding
    assert kwargs["prepend_student_bos"] is prepend_student_bos
    assert config.prepend_student_bos is prepend_student_bos
    assert (kwargs["model"], kwargs["n_bits"], kwargs["loss_mode"], kwargs["alpha"], kwargs["delta"]) == ("Qwen/Qwen3-4B-Base", 4, loss_type, 0.5, 4)
    assert "data_collator" not in kwargs
    assert kwargs["data_length"] == config.data_length
    assert train.build_sft_config.call_args.args[2] is train.AutoTokenizer.from_pretrained.return_value
    assert dataset.take.return_value.map.call_args.kwargs["fn_kwargs"] == {"n_bits": 4}
    train.AutoTokenizer.from_pretrained.assert_called_once_with("Qwen/Qwen3-4B-Base")
    trainer.return_value.train.assert_called_once_with()
    environment = {"WANDB_PROJECT": "old-project"}
    configure_wandb_environment(config, environment)
    assert environment["WANDB_PROJECT"] == "E20260916_qwen3_peft_convergence"


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--model", "Qwen/Qwen3-0.6B"),
        ("--model", "Qwen/Qwen3-8B-Base"),
        ("--loss-type", "kl"),
        ("--n-bits", "0"),
        ("--n-bits", "5"),
        ("--n-bits", "8"),
        ("--lr", "0"),
        ("--lr", "nan"),
        ("--alpha", "-1"),
        ("--alpha", "inf"),
        ("--delta", "-1"),
        ("--delta", "nan"),
        ("--min-gpt2-document-tokens", "-1"),
        ("--max-gpt2-document-tokens", "-1"),
        ("--min-qwen-document-tokens", "-1"),
        ("--max-qwen-document-tokens", "-1"),
    ],
)
def test_invalid_cli_fails_before_trainer(flag: str, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject malformed CLI settings before downloading or constructing a model."""
    build = Mock()
    monkeypatch.setattr(train, "build_trainer", build)
    result = CliRunner().invoke(train.main, [flag, value])
    assert result.exit_code != 0
    assert "Error" in result.output
    build.assert_not_called()


@pytest.mark.parametrize("overrides", [{"model": "other/Qwen3-4B-Base"}, {"model": "Qwen/Qwen3-1.7B"}, {"lr": float("inf")}, {"alpha": -1}, {"delta": float("nan")}, {"n_bits": 0}])
def test_python_interface_also_validates(overrides: dict[str, object]) -> None:
    """Prevent programmatic callers bypassing model/numeric validation in Click."""
    with pytest.raises(ValueError):
        train.experiment_config(**overrides)


def test_zero_weight_and_boost_are_usable_ablation_settings() -> None:
    """Accept zero ablations and distinguish their output paths from positive values."""
    zero = train.experiment_config(alpha=0, delta=0)
    positive = train.experiment_config(alpha=1, delta=2)
    assert zero.alpha == zero.delta == 0
    assert zero.run_name != positive.run_name
    assert zero.wandb_project == positive.wandb_project


@pytest.mark.parametrize("knob", ["lr", "alpha", "delta"])
def test_close_numeric_settings_keep_distinct_output_names(knob: str) -> None:
    """Prevent six-significant-digit display rounding from colliding checkpoint paths."""
    first = train.experiment_config(**{knob: 0.0003000001})
    second = train.experiment_config(**{knob: 0.0003000002})
    assert first.run_name != second.run_name


@pytest.mark.parametrize("tokenizer_name", ["gpt2", "qwen"])
def test_document_bounds_distinguish_runs_and_reject_reversed_ranges(tokenizer_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Partition active bounds into lower-only, zero-only, and reversed ranges."""
    prefix = f"{tokenizer_name}-"
    minimum_field = f"min_{prefix.replace('-', '_')}document_tokens"
    maximum_field = f"max_{prefix.replace('-', '_')}document_tokens"
    lower = train.experiment_config(**{minimum_field: 512})
    zero_only = train.experiment_config(**{minimum_field: 0, maximum_field: 0})
    unfiltered = train.experiment_config(min_gpt2_document_tokens=0, min_qwen_document_tokens=0)
    assert len({lower.run_name, zero_only.run_name, unfiltered.run_name}) == 3
    build = Mock()
    monkeypatch.setattr(train, "build_trainer", build)
    result = CliRunner().invoke(train.main, [f"--min-{prefix}document-tokens", "20", f"--max-{prefix}document-tokens", "10"])
    assert result.exit_code != 0
    build.assert_not_called()


def test_different_training_budgets_use_different_output_directories(monkeypatch, tmp_path) -> None:
    """Cover two valid budgets with otherwise identical flags; omit actual checkpoint IO."""
    from ciphers.kirchenbauer_et_al.src.configuration_kl_fineweb import build_sft_config

    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    configs = [train.experiment_config(num_training_tokens=budget) for budget in (4096 * 128 * 32, 4096 * 128 * 64)]
    tokenizer = Mock(return_value={"input_ids": [[1] * 25] * 2})
    outputs = []
    for config in configs:
        config.dtype, config.report_to = "float32", "none"
        outputs.append(build_sft_config(config, 1, tokenizer).output_dir)
    assert configs[0].max_steps != configs[1].max_steps
    assert outputs[0] != outputs[1]


@pytest.mark.parametrize("n_bits", [5, 8])
def test_experiment_python_interface_rejects_more_than_four_bits(n_bits: int) -> None:
    """Reject both the nearest unsupported width and the former eight-bit setting."""
    with pytest.raises(ValueError):
        train.experiment_config(n_bits=n_bits)


@pytest.mark.parametrize("global_batch_size,expected_steps", [(32, 1024), (64, 512), (128, 256)])
def test_shorter_experiment_budget_preserves_document_count_across_batches(global_batch_size: int, expected_steps: int) -> None:
    """Cover sweep/default batch sizes and the shared four-bit YAML budget.

    Steps adapt to batch size while the 33,554,432-token contract consumes
    32,768 training documents plus the disjoint validation reserve. No training
    or tokenization is performed; loader/Trainer wiring is tested separately.
    """
    from ciphers.kirchenbauer_et_al.src.configuration_kl_fineweb import load_training_config

    config = train.experiment_config(global_batch_size=global_batch_size)
    reference = load_training_config("ciphers/kirchenbauer_et_al/experiments/E20260912_qwen3_4b_4bit/config.yaml")
    assert config.max_steps == expected_steps
    assert config.max_steps * config.global_batch_size * config.data_length == 33_554_432
    assert reference.max_steps * reference.global_batch_size * reference.data_length == config.num_training_tokens
    assert config.validation_samples + config.max_steps * config.global_batch_size == 33_024
    assert config.min_qwen_document_tokens >= config.data_length
    assert reference.min_qwen_document_tokens >= reference.data_length


@pytest.mark.parametrize("child_fails", [False, True])
def test_sweep_environment_is_scoped(child_fails: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cover successful/failed children: route all runs without changing parent/cache roots."""
    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.delenv("STEGO_SWEEP_OUTPUT_DIR", raising=False)
    launch = Mock(side_effect=subprocess.CalledProcessError(1, "train") if child_fails else None)
    monkeypatch.setattr(run_sweep.subprocess, "run", launch)
    result = CliRunner().invoke(run_sweep.main)
    assert result.exit_code == int(child_fails), result.output
    assert launch.call_count == 48
    for call in launch.call_args_list:
        assert call.kwargs["env"]["STEGO_ARTIFACTS_DIR"] == str(tmp_path)
        assert call.kwargs["env"]["STEGO_SWEEP_OUTPUT_DIR"] == str(tmp_path / "E20260916_qwen3_0_6b_peft_convergence" / "sweep")
    assert os.environ["STEGO_ARTIFACTS_DIR"] == str(tmp_path)
    assert "STEGO_SWEEP_OUTPUT_DIR" not in os.environ
    assert not list(tmp_path.iterdir())


def test_sweep_without_artifacts_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run works without an artifacts root; a real launch fails before spawning children."""
    monkeypatch.delenv("STEGO_ARTIFACTS_DIR", raising=False)
    launch = Mock()
    monkeypatch.setattr(run_sweep.subprocess, "run", launch)
    dry_run = CliRunner().invoke(run_sweep.main, ["--dry-run"])
    assert dry_run.exit_code == 0, dry_run.output
    assert len(dry_run.output.splitlines()) == 48
    real_run = CliRunner().invoke(run_sweep.main)
    assert real_run.exit_code != 0
    assert "Set STEGO_ARTIFACTS_DIR" in real_run.output
    launch.assert_not_called()


@pytest.mark.parametrize("sweep", [False, True])
def test_output_override_preserves_cache_root(sweep: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cover direct/sweep launches: checkpoints and W&B move together; cache reads keep their root.

    Mock model, tokenizer, dataset, and Trainer IO; omit GPU training and W&B delivery.
    """
    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_PROJECT", "test")
    monkeypatch.setenv("WANDB_TAGS", "")
    output_root = tmp_path / "E20260916_qwen3_0_6b_peft_convergence" / "sweep" if sweep else tmp_path
    if sweep:
        monkeypatch.setenv("STEGO_SWEEP_OUTPUT_DIR", str(output_root))
    else:
        monkeypatch.delenv("STEGO_SWEEP_OUTPUT_DIR", raising=False)
    sft_args = SimpleNamespace(output_dir=str(tmp_path / "example-run"))
    monkeypatch.setattr(train, "build_sft_config", Mock(return_value=sft_args))
    monkeypatch.setattr(train.AutoTokenizer, "from_pretrained", Mock())
    cache_roots = []
    monkeypatch.setattr(train, "load_fineweb_cache", Mock(side_effect=lambda *args, **kwargs: cache_roots.append(os.environ["STEGO_ARTIFACTS_DIR"]) or Mock()))
    trainer = Mock()
    monkeypatch.setattr(train, "PrefixKLTrainer", trainer)
    train.build_trainer(train.experiment_config())
    expected_output = output_root / "example-run"
    assert trainer.call_args.kwargs["args"].output_dir == str(expected_output)
    assert os.environ["WANDB_DIR"] == str(expected_output)
    assert expected_output.is_dir()
    assert cache_roots == [str(tmp_path)]
