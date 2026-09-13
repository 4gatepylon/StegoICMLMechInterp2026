import pytest

from ciphers.kirchenbauer_et_al.src.configuration_kl_fineweb import (
    PrefixKLTrainingConfig,
    configure_wandb_environment,
    gradient_accumulation_steps,
    load_training_config,
    parse_args,
)
from wandb_archive import ARCHIVE_TAG

EIGHT_BIT_CONFIG_PATH = "ciphers/kirchenbauer_et_al/experiments/eight_bit_training_run.yaml"
OFFICIAL_CONFIG_PATHS = [f"ciphers/kirchenbauer_et_al/experiments/{bit_count}_bit_training_run.yaml" for bit_count in ("eight", "four", "two", "one")]


def test_optimization_knob_aliases() -> None:
    args = parse_args(["--lr", "0.001", "--batch-size", "4", "--grad-accum-steps", "8"])

    assert args.learning_rate == 0.001
    assert args.per_device_batch_size == 4
    assert gradient_accumulation_steps(args, world_size=2) == 8


def test_gradient_accumulation_defaults_to_global_batch_size() -> None:
    args = PrefixKLTrainingConfig(
        per_device_batch_size=2,
        gradient_accumulation_steps=None,
        global_batch_size=None,
    )

    assert gradient_accumulation_steps(args, world_size=4) == 4


def test_gradient_accumulation_rejects_conflicting_knobs() -> None:
    args = PrefixKLTrainingConfig(
        per_device_batch_size=2,
        gradient_accumulation_steps=4,
        global_batch_size=32,
    )

    with pytest.raises(ValueError, match="either global batch size or gradient accumulation steps"):
        gradient_accumulation_steps(args, world_size=1)


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_eight_bit_training_config_splits_global_batch_across_world_size(world_size: int) -> None:
    """Cover exact distributed batch splits; unsupported world sizes and model execution are omitted."""
    config = load_training_config(EIGHT_BIT_CONFIG_PATH)

    gradient_accumulation = gradient_accumulation_steps(config, world_size)

    assert config.per_device_batch_size * world_size * gradient_accumulation == config.global_batch_size


def test_command_line_overrides_eight_bit_config() -> None:
    """Cover precedence for representative numeric fields; other field types are omitted."""
    config = parse_args(["--config", EIGHT_BIT_CONFIG_PATH, "--lr", "0.001", "--save-steps", "128"])

    assert config.learning_rate == 0.001
    assert config.save_steps == 128
    assert config.max_steps == 1024


def test_prefix_kl_training_config_rejects_unknown_fields() -> None:
    """Cover typo rejection at the schema boundary; malformed YAML syntax is out of scope."""
    with pytest.raises(ValueError, match="extra_forbidden"):
        PrefixKLTrainingConfig.model_validate({"unexpected_setting": True})


def test_official_training_configs_add_archive_tag_while_default_runs_remain_unmarked() -> None:
    """Cover every launcher YAML and the unconfigured partition; W&B initialization is omitted."""
    environment = {"WANDB_TAGS": "manual-tag"}
    for config_path in OFFICIAL_CONFIG_PATHS:
        configure_wandb_environment(load_training_config(config_path), environment)

    assert environment["WANDB_TAGS"].split(",") == ["manual-tag", ARCHIVE_TAG]

    unmarked_environment: dict[str, str] = {}
    configure_wandb_environment(PrefixKLTrainingConfig(), unmarked_environment)
    assert "WANDB_TAGS" not in unmarked_environment
