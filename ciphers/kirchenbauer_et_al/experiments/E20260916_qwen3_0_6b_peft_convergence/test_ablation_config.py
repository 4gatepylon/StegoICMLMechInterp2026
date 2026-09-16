"""Exercise sweep commands, invalid inputs, naming, and trainer wiring on CPU.

Partitions include all three models/four bit lengths, both loss types, zero and
positive objective weights, invalid models, and nonpositive/nonfinite numbers.
Document filtering covers CLI forwarding of nested bounds, invalid ranges,
the selected tokenizer, and distinct filtered-run output names.
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

from ciphers.kirchenbauer_et_al.experiments.E20260916_qwen3_0_6b_peft_convergence import train
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
        [
            "--lr",
            "0.001",
            "--model-name",
            "Qwen/Qwen3-4B-Base",
            "--n-bits",
            "8",
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
    assert config.run_name == f"qwen3-4b-8bit-lr0.001-gb128-{loss_type}-a0.5-d4-tokens{config.num_training_tokens}-gpt2-100-2000-qwen-512-2048"
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
    zero_only = train.experiment_config(**{maximum_field: 0})
    unfiltered = train.experiment_config()
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
