import os
import runpy
import subprocess
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

EXPERIMENT_PATH = "ciphers/kirchenbauer_et_al/experiments/E20260912_qwen3_4b_4bit"
TRAINING_CONFIG_PATH = f"{EXPERIMENT_PATH}/config.yaml"


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
def test_yaml_training_config_splits_global_batch_across_world_size(world_size: int) -> None:
    """Cover exact distributed batch splits; unsupported world sizes and model execution are omitted."""
    config = load_training_config(TRAINING_CONFIG_PATH)

    gradient_accumulation = gradient_accumulation_steps(config, world_size)

    assert config.per_device_batch_size * world_size * gradient_accumulation == config.global_batch_size


def test_command_line_overrides_yaml_config() -> None:
    """Cover precedence for representative numeric fields; other field types are omitted."""
    config = parse_args(
        [
            "--config",
            TRAINING_CONFIG_PATH,
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
    assert config.max_steps == 256
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


def test_prefix_kl_training_config_rejects_unknown_fields() -> None:
    """Cover typo rejection at the schema boundary; malformed YAML syntax is out of scope."""
    with pytest.raises(ValueError, match="extra_forbidden"):
        PrefixKLTrainingConfig.model_validate({"unexpected_setting": True})


def test_configured_archive_tag_is_preserved_without_marking_default_runs() -> None:
    """Cover tag merging, repeated application, and untagged runs; W&B initialization is omitted."""
    environment = {"WANDB_TAGS": "manual-tag"}
    config = PrefixKLTrainingConfig(wandb_tags=[ARCHIVE_TAG])
    configure_wandb_environment(config, environment)
    configure_wandb_environment(config, environment)

    assert environment["WANDB_TAGS"].split(",") == ["manual-tag", ARCHIVE_TAG]

    unmarked_environment: dict[str, str] = {}
    configure_wandb_environment(PrefixKLTrainingConfig(), unmarked_environment)
    assert "WANDB_TAGS" not in unmarked_environment


@pytest.mark.parametrize("reject_padding", [True, False])
def test_document_filter_yaml_cli_and_training_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reject_padding: bool) -> None:
    """Cover YAML bounds, CLI overrides/unlimited, and forwarding before splitting.

    Cache loading, tokenizer, trainer, and training configuration construction
    are mocked: no network, model execution, or GPU is used.
    """
    from ciphers.kirchenbauer_et_al.src import cache_fineweb, configuration_kl_fineweb

    monkeypatch.setattr(configuration_kl_fineweb, "REPO_ROOT", tmp_path)
    (tmp_path / "filter.yaml").write_text("min_gpt2_document_tokens: 100\nmax_gpt2_document_tokens: 500\nmin_qwen_document_tokens: 90\nmax_qwen_document_tokens: 490\n")
    config = parse_args(["--config", "filter.yaml"])
    assert (config.min_gpt2_document_tokens, config.max_gpt2_document_tokens) == (100, 500)
    assert (config.min_qwen_document_tokens, config.max_qwen_document_tokens) == (90, 490)
    config = parse_args(
        [
            "--config",
            "filter.yaml",
            "--min-gpt2-document-tokens",
            "200",
            "--max-gpt2-document-tokens",
            "none",
            "--min-qwen-document-tokens",
            "180",
            "--max-qwen-document-tokens",
            "none",
        ]
    )
    assert (config.min_gpt2_document_tokens, config.max_gpt2_document_tokens) == (200, None)
    assert (config.min_qwen_document_tokens, config.max_qwen_document_tokens) == (180, None)
    config = parse_args(
        [
            "--config",
            "filter.yaml",
            "--min-gpt2-document-tokens",
            "200",
            "--max-gpt2-document-tokens",
            "none",
            "--min-qwen-document-tokens",
            "180",
            "--max-qwen-document-tokens",
            "none",
            "--reject-document-padding" if reject_padding else "--no-reject-document-padding",
        ]
    )
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

    tokenizer = sys.modules["transformers"].AutoTokenizer.from_pretrained.return_value
    cache_loader.assert_called_once_with(
        config.dataset_cache_name,
        minimum_documents=config.validation_samples + config.max_steps * 32,
        min_gpt2_document_tokens=200,
        max_gpt2_document_tokens=None,
        min_qwen_document_tokens=180,
        max_qwen_document_tokens=None,
        tokenizer=tokenizer,
    )
    cache_loader.return_value.take.assert_called_once_with(config.validation_samples)
    cache_loader.return_value.skip.assert_called_once_with(config.validation_samples)
    sft_config_builder.assert_called_once_with(config, 32, tokenizer)
    trainer_kwargs = sys.modules["ciphers.kirchenbauer_et_al.src.trainer_kl_fineweb"].PrefixKLTrainer.call_args.kwargs
    assert trainer_kwargs["data_collator"].keywords["data_length"] == config.data_length
    assert trainer_kwargs["args"] is sft_config_builder.return_value
    assert trainer_kwargs["reject_document_padding"] is reject_padding
    assert trainer_kwargs["dump_inputs"] == config.dump_inputs


@pytest.mark.parametrize("rank", ["0", "1"])
@pytest.mark.parametrize("n_bits,prefix_length", [(8, 32), (4, 28), (2, 26), (1, 25)])
def test_data_budget_reaches_trainer_logging_and_input_token_count(n_bits, prefix_length, rank, monkeypatch, tmp_path, capsys) -> None:
    """Cover the retained YAML with 1/2/4/8-bit CLI overrides, both ranks, and total-input token accounting.

    Prefix widths are supplied by a tokenizer stub. External logging, full
    training, and distributed execution are omitted; rank selection is simulated.
    """
    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", rank)
    config = parse_args(["--config", TRAINING_CONFIG_PATH, "--n-bits", str(n_bits), "--data-length", "2048", "--report-to", "none", "--dtype", "float32"])
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
    from ciphers.kirchenbauer_et_al.experiments.E20260916_qwen3_0_6b_peft_convergence.train import experiment_config

    config = experiment_config(local_batch_size=4, global_batch_size=global_batch_size)
    assert config.max_steps * config.data_length * global_batch_size == config.num_training_tokens
    assert config.save_total_limit == (config.max_steps + config.save_steps - 1) // config.save_steps
    with pytest.raises(ValueError, match="must be divisible by data length"):
        experiment_config(num_training_tokens=config.num_training_tokens + 1)


def test_convergence_entrypoint_wires_data_and_total_lengths(monkeypatch, tmp_path) -> None:
    """Cover experiment startup to Trainer arguments; dataset/model loading and training are mocked."""
    from ciphers.kirchenbauer_et_al.experiments.E20260916_qwen3_0_6b_peft_convergence import train

    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    config = train.experiment_config(dump_inputs=3)
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
    assert kwargs["dump_inputs"] == 3


@pytest.mark.parametrize("flag", ["--dump-inputs", "--dump_inputs"])
@pytest.mark.parametrize("value", ["0", "3", "-1", "1.5"])
def test_dump_limit_cli_validation_and_forwarding(flag, value, monkeypatch) -> None:
    """Cover both CLI spellings, disabled/positive limits and negative/fractional errors.

    Exercise argparse and Click through config construction; model construction
    is mocked. Dump contents and lifecycle behavior are tested with the KL loss.
    """
    from click.testing import CliRunner

    from ciphers.kirchenbauer_et_al.experiments.E20260916_qwen3_0_6b_peft_convergence import train

    build = Mock()
    monkeypatch.setattr(train, "build_trainer", build)
    result = CliRunner().invoke(train.main, [flag, value])
    if value in {"-1", "1.5"}:
        assert result.exit_code != 0
        build.assert_not_called()
        with pytest.raises((ValueError, SystemExit)):
            parse_args([flag, value])
    else:
        assert result.exit_code == 0, result.output
        assert build.call_args.args[0].dump_inputs == int(value)
        assert parse_args([flag, value]).dump_inputs == int(value)


def test_dump_limit_yaml_override(tmp_path, monkeypatch) -> None:
    """Cover persisted limits, explicit disable overriding YAML, and invalid YAML."""
    from ciphers.kirchenbauer_et_al.src import configuration_kl_fineweb

    monkeypatch.setattr(configuration_kl_fineweb, "REPO_ROOT", tmp_path)
    path = tmp_path / "dump.yaml"
    path.write_text("dump_inputs: 3\n")
    assert parse_args(["--config", "dump.yaml"]).dump_inputs == 3
    assert parse_args(["--config", "dump.yaml", "--dump-inputs", "0"]).dump_inputs == 0
    path.write_text("dump_inputs: -1\n")
    with pytest.raises(ValueError, match="dump_inputs"):
        load_training_config("dump.yaml")


@pytest.mark.parametrize("exit_code", [0, 23])
def test_original_run_launcher(tmp_path: Path, exit_code: int) -> None:
    """Cover one successful/failed launch from outside the repo using its real YAML.

    Fake Conda records arguments and cwd; model/data loading, GPU execution,
    Conda environment activation, and live W&B delivery are omitted.
    """
    repo_root = Path(__file__).resolve().parents[1]
    fake_conda = tmp_path / "conda"
    fake_conda.write_text('#!/usr/bin/env bash\npwd > "$LAUNCH_LOG"\nprintf "%s\\n" "$@" >> "$LAUNCH_LOG"\nexit "$LAUNCH_EXIT_CODE"\n')
    fake_conda.chmod(0o755)
    log = tmp_path / "launch.log"
    environment = os.environ.copy()
    environment.update(PATH=f"{tmp_path}:{environment['PATH']}", LAUNCH_LOG=str(log), LAUNCH_EXIT_CODE=str(exit_code))

    result = subprocess.run(["bash", str(repo_root / EXPERIMENT_PATH / "run.sh")], cwd=tmp_path, env=environment, capture_output=True, text=True)

    assert result.returncode == exit_code, result.stderr
    working_directory, *arguments = log.read_text().splitlines()
    assert Path(working_directory) == repo_root
    assert arguments == [
        "run",
        "--no-capture-output",
        "-n",
        "stego",
        "python",
        "-m",
        "ciphers.kirchenbauer_et_al.src.train_kl_fineweb",
        "--config",
        TRAINING_CONFIG_PATH,
        "--wandb-project",
        "stego-kirchenbauer-prefix-kl",
    ]
    config = parse_args(arguments[7:])
    assert (config.model, config.n_bits) == ("Qwen/Qwen3-4B-Base", 4)


def test_padding_guard_yaml_can_be_overridden_by_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise both YAML Boolean values and CLI precedence; no Trainer is created."""
    from ciphers.kirchenbauer_et_al.src import configuration_kl_fineweb

    monkeypatch.setattr(configuration_kl_fineweb, "REPO_ROOT", tmp_path)
    (tmp_path / "padding.yaml").write_text("reject_document_padding: false\n")
    assert not parse_args(["--config", "padding.yaml"]).reject_document_padding
    assert parse_args(["--config", "padding.yaml", "--reject-document-padding"]).reject_document_padding
    (tmp_path / "padding.yaml").write_text("reject_document_padding: true\n")
    assert not parse_args(["--config", "padding.yaml", "--no-reject-document-padding"]).reject_document_padding


@pytest.mark.parametrize("mode", ["token", "char"])
def test_removed_concatenation_setting_is_rejected(mode) -> None:
    """Reject both historical mode values through config and CLI; YAML uses the same schema."""
    with pytest.raises(ValueError, match="concatenation_space"):
        PrefixKLTrainingConfig.model_validate({"concatenation_space": mode})
    with pytest.raises(SystemExit):
        parse_args(["--concatenation-space", mode])
