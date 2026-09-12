import pytest

from ciphers.kirchenbauer_et_al.binary_classification_mvp.train_kl_fineweb import (
    TrainingConfig,
    gradient_accumulation_steps,
    load_training_config,
    parse_args,
)

OFFICIAL_CONFIG_PATH = "ciphers/kirchenbauer_et_al/experiments/official_training_run.yaml"


def test_optimization_knob_aliases() -> None:
    args = parse_args(["--lr", "0.001", "--batch-size", "4", "--grad-accum-steps", "8"])

    assert args.learning_rate == 0.001
    assert args.per_device_batch_size == 4
    assert gradient_accumulation_steps(args, world_size=2) == 8


def test_gradient_accumulation_defaults_to_global_batch_size() -> None:
    args = TrainingConfig(
        per_device_batch_size=2,
        gradient_accumulation_steps=None,
        global_batch_size=None,
    )

    assert gradient_accumulation_steps(args, world_size=4) == 4


def test_gradient_accumulation_rejects_conflicting_knobs() -> None:
    args = TrainingConfig(
        per_device_batch_size=2,
        gradient_accumulation_steps=4,
        global_batch_size=32,
    )

    with pytest.raises(ValueError, match="either global batch size or gradient accumulation steps"):
        gradient_accumulation_steps(args, world_size=1)


def test_official_training_config() -> None:
    """Cover the one official run; model execution and distributed launch are out of scope."""
    config = load_training_config(OFFICIAL_CONFIG_PATH)

    assert config.max_steps == 1024
    assert config.global_batch_size == 128
    assert config.per_device_batch_size == 2
    assert config.learning_rate == 3e-4
    assert config.save_steps == 256
    assert config.save_total_limit == 4
    assert gradient_accumulation_steps(config, world_size=8) == 8


def test_command_line_overrides_official_config() -> None:
    """Cover precedence for representative numeric fields; other field types are omitted."""
    config = parse_args(["--config", OFFICIAL_CONFIG_PATH, "--lr", "0.001", "--save-steps", "128"])

    assert config.learning_rate == 0.001
    assert config.save_steps == 128
    assert config.max_steps == 1024


def test_training_config_rejects_unknown_fields() -> None:
    """Cover typo rejection at the schema boundary; malformed YAML syntax is out of scope."""
    with pytest.raises(ValueError, match="extra_forbidden"):
        TrainingConfig.model_validate({"unexpected_setting": True})
