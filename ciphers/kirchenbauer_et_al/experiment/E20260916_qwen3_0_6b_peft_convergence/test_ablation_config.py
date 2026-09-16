"""Exercise sweep commands, invalid inputs, naming, and trainer wiring on CPU.

Partitions include all three models/four bit lengths, both loss types, zero and
positive objective weights, invalid models, and nonpositive/nonfinite numbers.
Trainer, tokenizer, dataset, and SFT construction are mocked: GPU training,
model downloads, distributed execution, and live W&B delivery are omitted.
"""

import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import get_args
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from ciphers.kirchenbauer_et_al.experiment.E20260916_qwen3_0_6b_peft_convergence import train
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
    assert f"{config.n_bits}bit-lr0.0003-gb32" in config.run_name
    build.return_value.train.assert_called_once_with()


def test_sweep_covers_full_cartesian_product() -> None:
    """Catch missing/duplicated commands in the documented model × bits sweep."""
    combinations = [(args[args.index("--model") + 1], int(args[args.index("--n-bits") + 1])) for args in SWEEP_COMMANDS]
    assert len(combinations) == len(set(combinations))
    assert set(combinations) == {(model, bits) for model in get_args(train.QwenModel) for bits in (1, 2, 4, 8)}
    run_names = [train.experiment_config(local_batch_size=2, global_batch_size=32, model=model, n_bits=bits).run_name for model, bits in combinations]
    assert len(set(run_names)) == len(combinations)


@pytest.mark.parametrize("loss_type", ["nll", "ignore_prefix"])
def test_objective_overrides_reach_trainer(loss_type: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        ["--lr", "0.001", "--model-name", "Qwen/Qwen3-4B-Base", "--n-bits", "8", "--loss-type", loss_type, "--alpha", "0.5", "--delta", "4"],
    )
    assert result.exit_code == 0, result.output
    config = train.build_sft_config.call_args.args[0]
    assert config.learning_rate == 0.001
    assert config.run_name == f"qwen3-4b-8bit-lr0.001-gb128-{loss_type}-a0.5-d4"
    kwargs = trainer.call_args.kwargs
    assert (kwargs["model"], kwargs["n_bits"], kwargs["loss_mode"], kwargs["alpha"], kwargs["delta"]) == ("Qwen/Qwen3-4B-Base", 8, loss_type, 0.5, 4)
    assert kwargs["data_collator"].keywords["n_bits"] == 8
    assert kwargs["data_collator"].keywords["data_length"] == config.data_length
    assert train.build_sft_config.call_args.args[2] is train.AutoTokenizer.from_pretrained.return_value
    assert dataset.take.return_value.map.call_args.kwargs["fn_kwargs"] == {"n_bits": 8}
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
        ("--lr", "0"),
        ("--lr", "nan"),
        ("--alpha", "-1"),
        ("--alpha", "inf"),
        ("--delta", "-1"),
        ("--delta", "nan"),
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
