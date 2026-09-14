import runpy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from ciphers.kirchenbauer_et_al.src.configuration_kl_fineweb import (
    PrefixKLTrainingConfig,
    build_sft_config,
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
    config = parse_args(
        [
            "--config",
            EIGHT_BIT_CONFIG_PATH,
            "--lr",
            "0.001",
            "--save-steps",
            "128",
            "--include-num-input-tokens-seen",
            "all",
        ]
    )

    assert config.learning_rate == 0.001
    assert config.save_steps == 128
    assert config.max_steps == 1024
    assert config.include_num_input_tokens_seen == "all"


@pytest.mark.parametrize("token_count_mode", ["all", "non_padding", "no"])
def test_command_line_accepts_transformers_token_count_modes(token_count_mode: str) -> None:
    """Cover every supported native counting mode; Trainer execution and W&B delivery are omitted."""
    config = parse_args(["--include-num-input-tokens-seen", token_count_mode])

    assert config.include_num_input_tokens_seen == token_count_mode


def test_command_line_rejects_unknown_token_count_mode() -> None:
    """Cover invalid CLI input at the parser boundary; equivalent malformed YAML is omitted."""
    with pytest.raises(SystemExit):
        parse_args(["--include-num-input-tokens-seen", "sometimes"])


def test_token_count_mode_reaches_sft_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cover configuration-to-Trainer wiring; model execution and W&B delivery are omitted."""
    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    config = PrefixKLTrainingConfig(include_num_input_tokens_seen="all", report_to="none")

    training_arguments = build_sft_config(config, grad_accumulation_steps=8)

    assert training_arguments.include_num_input_tokens_seen == "all"


def test_official_configs_enable_non_padding_token_counting() -> None:
    """Cover token-count configuration in every launcher YAML; training and W&B delivery are omitted."""
    assert all(load_training_config(config_path).include_num_input_tokens_seen == "non_padding" for config_path in OFFICIAL_CONFIG_PATHS)


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


def test_document_filter_yaml_cli_and_training_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover YAML bounds, CLI overrides/unlimited, and forwarding before splitting.

    Cache loading, tokenizer, trainer, and training configuration construction
    are mocked: no network, model execution, or GPU is used.
    """
    from ciphers.kirchenbauer_et_al.src import cache_fineweb, configuration_kl_fineweb

    monkeypatch.setattr(configuration_kl_fineweb, "REPO_ROOT", tmp_path)
    (tmp_path / "filter.yaml").write_text("min_document_tokens: 100\nmax_document_tokens: 500\n")
    config = parse_args(["--config", "filter.yaml"])
    assert (config.min_document_tokens, config.max_document_tokens) == (100, 500)
    config = parse_args(["--config", "filter.yaml", "--min-document-tokens", "200", "--max-document-tokens", "none"])
    assert (config.min_document_tokens, config.max_document_tokens) == (200, None)
    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(configuration_kl_fineweb, "parse_args", lambda: config)
    cache_loader = Mock()
    monkeypatch.setattr(cache_fineweb, "load_fineweb_cache", cache_loader)
    for module in (
        "torch",
        "peft",
        "transformers",
        "trl",
        "ciphers.kirchenbauer_et_al.src.data_kl_fineweb",
        "ciphers.kirchenbauer_et_al.src.trainer_kl_fineweb",
    ):
        monkeypatch.setitem(sys.modules, module, Mock())

    runpy.run_module("ciphers.kirchenbauer_et_al.src.train_kl_fineweb", run_name="__main__")

    cache_loader.assert_called_once_with(
        config.dataset_cache_name,
        minimum_documents=config.validation_samples + config.max_steps * 32,
        min_document_tokens=200,
        max_document_tokens=None,
    )
    cache_loader.return_value.take.assert_called_once_with(config.validation_samples)
    cache_loader.return_value.skip.assert_called_once_with(config.validation_samples)
