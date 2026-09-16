import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from trl import SFTTrainer

from ciphers.kirchenbauer_et_al.src.configuration_kl_fineweb import (
    PrefixKLTrainingConfig,
    build_sft_config,
    configure_wandb_environment,
    gradient_accumulation_steps,
    load_training_config,
    parse_args,
)
from ciphers.kirchenbauer_et_al.src.trainer_kl_fineweb import PrefixKLTrainer
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

    tokenizer = Mock(return_value={"input_ids": [[1] * 32, [1] * 32]})
    training_arguments = build_sft_config(config, grad_accumulation_steps=8, tokenizer=tokenizer)

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
    sft_config_builder = Mock()
    monkeypatch.setattr(configuration_kl_fineweb, "build_sft_config", sft_config_builder)
    cache_loader = Mock()
    monkeypatch.setattr(cache_fineweb, "load_fineweb_cache", cache_loader)
    for module in (
        "peft",
        "transformers",
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
    tokenizer = sys.modules["transformers"].AutoTokenizer.from_pretrained.return_value
    sft_config_builder.assert_called_once_with(config, 32, tokenizer)
    trainer_kwargs = sys.modules["ciphers.kirchenbauer_et_al.src.trainer_kl_fineweb"].PrefixKLTrainer.call_args.kwargs
    assert trainer_kwargs["data_collator"].keywords["data_length"] == config.data_length
    assert trainer_kwargs["args"] is sft_config_builder.return_value


@pytest.mark.parametrize("rank", ["0", "1"])
@pytest.mark.parametrize("config_path,prefix_length", list(zip(OFFICIAL_CONFIG_PATHS, [32, 28, 26, 25])))
def test_data_budget_reaches_trainer_logging_and_input_token_count(config_path, prefix_length, rank, monkeypatch, tmp_path, capsys) -> None:
    """Cover every YAML, CLI override, total-length wiring, rank-zero output, and token accounting.

    Prefix widths are supplied by a tokenizer stub. External logging, full
    training, and distributed execution are omitted; rank selection is simulated.
    """
    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", rank)
    config = parse_args(["--config", config_path, "--data-length", "2048", "--report-to", "none", "--dtype", "float32"])
    tokenizer = Mock(return_value={"input_ids": [[1] * prefix_length] * 2})
    training_args = build_sft_config(config, 2, tokenizer)
    assert config.data_length == 2048
    assert training_args.max_length == 2048 + prefix_length
    output = capsys.readouterr().out
    if rank == "0":
        assert f"data_length=2048 + prefix_length={prefix_length} = max_length={2048 + prefix_length}" in output
        assert f"{config.n_bits}-bit secret message" in output
        assert f"{2048 // config.n_bits} data positions per bit" in output
    else:
        assert output == ""
    trainer = object.__new__(PrefixKLTrainer)
    trainer.state = SimpleNamespace(global_step=3)
    trainer.args = training_args
    trainer.accelerator = SimpleNamespace(num_processes=1)
    logs = {}
    with patch.object(SFTTrainer, "log"):
        PrefixKLTrainer.log(trainer, logs)
    assert logs["num_padded_input_tokens_seen"] == 3 * (2048 + prefix_length) * config.per_device_batch_size * 2


@pytest.mark.parametrize("data_length", [0, -8, 4095])
def test_config_rejects_invalid_data_budget(data_length) -> None:
    """Cover nonpositive and nondivisible budgets at schema construction; tokenizer checks are separate."""
    with pytest.raises(ValueError, match="data_length"):
        PrefixKLTrainingConfig(data_length=data_length, n_bits=8)


def test_legacy_total_length_setting_is_rejected(tmp_path, monkeypatch) -> None:
    """Cover old YAML and CLI names, preventing silent changes to historical run semantics."""
    from ciphers.kirchenbauer_et_al.src import configuration_kl_fineweb

    monkeypatch.setattr(configuration_kl_fineweb, "REPO_ROOT", tmp_path)
    (tmp_path / "legacy.yaml").write_text("max_length: 4096\n")
    with pytest.raises(ValueError, match="max_length"):
        load_training_config("legacy.yaml")
    with pytest.raises(SystemExit):
        parse_args(["--max-length", "4096"])


@pytest.mark.parametrize("global_batch_size", [32, 128])
def test_convergence_experiment_budget_excludes_prefix(global_batch_size) -> None:
    """Cover both documented batch sizes and indivisible budgets; training and checkpoints are omitted."""
    from ciphers.kirchenbauer_et_al.experiment.E20260916_qwen3_0_6b_peft_convergence.train import experiment_config

    config = experiment_config(local_batch_size=4, global_batch_size=global_batch_size)
    assert config.max_steps * config.data_length * global_batch_size == config.num_training_tokens
    assert config.save_total_limit == (config.max_steps + config.save_steps - 1) // config.save_steps
    with pytest.raises(ValueError, match="must be divisible by data length"):
        experiment_config(num_training_tokens=config.num_training_tokens + 1)


def test_convergence_entrypoint_wires_data_and_total_lengths(monkeypatch, tmp_path) -> None:
    """Cover experiment startup to Trainer arguments; dataset/model loading and training are mocked."""
    from ciphers.kirchenbauer_et_al.experiment.E20260916_qwen3_0_6b_peft_convergence import train

    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    config = train.experiment_config()
    config.dtype, config.report_to, config.wandb_tags = "float32", "none", []
    tokenizer = Mock(return_value={"input_ids": [[1] * 25] * 2})
    monkeypatch.setattr(train.AutoTokenizer, "from_pretrained", Mock(return_value=tokenizer))
    monkeypatch.setattr(train, "load_fineweb_cache", Mock())
    trainer_constructor = Mock()
    monkeypatch.setattr(train, "PrefixKLTrainer", trainer_constructor)
    assert train.build_trainer(config) is trainer_constructor.return_value
    kwargs = trainer_constructor.call_args.kwargs
    assert kwargs["data_collator"].keywords["data_length"] == config.data_length
    assert kwargs["args"].max_length == config.data_length + 25
    assert kwargs["args"].save_only_model
