"""Token schedule partitions: global batches 16/32/64/128 and varying world size;
short/full sequence widths; intervals below/on/above step boundaries; zero warmup;
invalid intervals and indivisible total budgets; partial final save intervals.
CLI/Trainer wiring uses mocks for data, tokenizer and model; real SFTConfig runs on
CPU. Omit GPU/distributed execution, live downloads/W&B and convergence claims.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from ciphers.kirchenbauer_et_al.experiment.E20260916_qwen3_0_6b_peft_convergence import train


@pytest.mark.parametrize("global_batch", [16, 32, 64, 128])
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_smaller_batches_preserve_schedule_token_counts(global_batch, world_size):
    config = train.experiment_config(local_batch_size=4, global_batch_size=global_batch)
    accumulation = train.gradient_accumulation_steps(config, world_size)
    tokens_per_step = config.per_device_batch_size * world_size * accumulation * config.max_length
    assert config.max_steps * tokens_per_step == config.num_training_tokens
    for field in ("warmup_steps", "eval_steps", "save_steps", "logging_steps"):
        assert getattr(config, field) * tokens_per_step == getattr(config, f"{field}_in_tokens")
    assert config.save_total_limit * config.save_steps * tokens_per_step == config.num_training_tokens


@pytest.mark.parametrize("max_length", [2048, 4096])
def test_sequence_length_changes_derived_steps(max_length):
    config = train.FixedBudgetTrainingConfig(global_batch_size=128, max_length=max_length)
    baseline = train.experiment_config()
    for field in ("max_steps", "warmup_steps", "eval_steps", "save_steps", "logging_steps"):
        assert getattr(config, field) * max_length == getattr(baseline, field) * baseline.max_length


@pytest.mark.parametrize("warmup_tokens", [0, 1, 32, 33])
def test_rounding_and_partial_interval_retention(warmup_tokens):
    config = train.FixedBudgetTrainingConfig(
        global_batch_size=4,
        max_length=8,
        num_training_tokens=320,
        warmup_steps_in_tokens=warmup_tokens,
        eval_steps_in_tokens=33,
        save_steps_in_tokens=97,
        logging_steps_in_tokens=1,
    )
    tokens_per_step = config.global_batch_size * config.max_length
    for field in ("warmup_steps", "eval_steps", "save_steps", "logging_steps"):
        requested = getattr(config, f"{field}_in_tokens")
        assert 0 <= getattr(config, field) * tokens_per_step - requested < tokens_per_step
    assert (config.eval_steps, config.save_steps, config.logging_steps) == (2, 4, 1)
    assert config.save_total_limit == len(range(config.save_steps, config.max_steps, config.save_steps)) + 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("warmup_steps_in_tokens", -1),
        ("eval_steps_in_tokens", 0),
        ("save_steps_in_tokens", 0),
        ("logging_steps_in_tokens", 0),
        ("num_training_tokens", 33),
    ],
)
def test_invalid_token_schedule_is_rejected(field, value):
    with pytest.raises(ValidationError):
        train.FixedBudgetTrainingConfig(global_batch_size=4, max_length=8, **{field: value})


def test_cli_and_trainer_consume_scaled_schedule(monkeypatch, tmp_path):
    """A smaller global batch reaches real Trainer args and preserves data demand."""
    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_TAGS", "")
    original_config = train.experiment_config

    def cpu_config(*args, **kwargs):
        config = original_config(*args, **kwargs)
        config.dtype, config.report_to = "float32", "none"
        return config

    monkeypatch.setattr(train, "experiment_config", cpu_config)
    cache_loader, trainer = Mock(), Mock()
    monkeypatch.setattr(train, "load_fineweb_cache", cache_loader)
    monkeypatch.setattr(train, "PrefixKLTrainer", trainer)
    monkeypatch.setattr(train.AutoTokenizer, "from_pretrained", lambda _: SimpleNamespace(pad_token=None, eos_token="eos"))
    result = CliRunner().invoke(train.main, ["--local-batch-size", "4", "--global-batch-size", "32"])
    assert result.exit_code == 0, result.output
    args = trainer.call_args.kwargs["args"]
    assert (args.max_steps, args.warmup_steps, args.eval_steps, args.save_steps, args.logging_steps) == (4096, 200, 16, 128, 4)
    assert args.save_only_model and args.save_total_limit == 32
    assert cache_loader.call_args.kwargs["minimum_documents"] == 256 + train.NUM_TRAINING_TOKENS // train.SEQUENCE_LENGTH
    trainer.return_value.train.assert_called_once_with()
