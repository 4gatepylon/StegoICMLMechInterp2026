"""JSON/YAML experiment configuration and CLI-default resolution."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from peft import LoraConfig
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from trl import SFTConfig

from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.artifacts import artifact_paths
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.constants import MODEL_NAME, REPO_ROOT


class ModelSettings(BaseModel):
    """Hugging Face IDs or explicitly base-relative local paths."""

    model_config = ConfigDict(extra="forbid")

    weights: str | None = MODEL_NAME
    weights_path: str | None = None
    config: dict[str, Any] | None = None
    config_path: str | None = None
    tokenizer: str | None = MODEL_NAME
    tokenizer_path: str | None = None
    initialization_seed: int = 42

    @model_validator(mode="after")
    def validate_sources(self) -> ModelSettings:
        if self.weights is not None and self.weights_path is not None:
            raise ValueError("Set only one of model.weights and model.weights_path")
        if self.config is not None and self.config_path is not None:
            raise ValueError("Set only one of model.config and model.config_path")
        if self.tokenizer is not None and self.tokenizer_path is not None:
            raise ValueError("Set only one of model.tokenizer and model.tokenizer_path")
        if self.weights is None and self.weights_path is None:
            if self.config is None and self.config_path is None:
                raise ValueError("Random initialization requires model.config or config_path")
        if self.tokenizer is None and self.tokenizer_path is None:
            raise ValueError("Set model.tokenizer or model.tokenizer_path")
        return self


class DataSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prefix_train_tokens: int = 65_536
    encoding_train_tokens: int = 65_536
    validation_tokens: int = 16_384
    generation_prompts: int = 256
    sequence_length: int = 256
    generation_prompt_length: int = 128
    min_sequence_length: int = 32
    dataset_seed: int = 42
    shuffle_buffer_size: int = 10_000


class DistillationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delta: float = Field(default=2.0, gt=0)
    vocab_seed: int = 42
    max_eval_sequences: int = Field(default=16, gt=0)


LORA_DEFAULTS = {
    "task_type": "CAUSAL_LM",
    "target_modules": "all-linear",
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
}

TRAINER_DEFAULTS = {
    "output_dir": "trainer_output",  # Replaced with the stage artifact path.
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "num_train_epochs": 1,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "constant",
    "logging_first_step": True,
    "logging_steps": 8,
    "eval_strategy": "steps",
    "eval_steps": 32,
    "save_strategy": "no",
    "seed": 1234,
    "report_to": "none",
    "bf16": False,
    "fp16": False,
    "max_length": None,
    "packing": False,
    "padding_free": False,
    "completion_only_loss": False,
    "loss_type": "nll",
    "use_liger_kernel": False,
}


def default_lora_config() -> LoraConfig:
    return LoraConfig(**LORA_DEFAULTS)


def default_trainer_config() -> SFTConfig:
    """The complete built-in training configuration, using TRL's own type."""

    return SFTConfig(**TRAINER_DEFAULTS)


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str = "auto"
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"


class GenerationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sampling_seed: int = 2026
    temperature: float = 0.7
    max_new_tokens: int = 200


class WandbSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["disabled", "online", "offline"] = "disabled"
    project: str = "stego-kirchenbauer-binary-classification"
    run_name: str = "qwen3-4b-base-fineweb-64k"
    entity: str | None = None

    @model_validator(mode="after")
    def validate_names(self) -> WandbSettings:
        if self.mode != "disabled" and (not self.project.strip() or not self.run_name.strip()):
            raise ValueError("wandb.project and wandb.run_name must be non-empty when W&B is enabled")
        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths_relative_to: Literal["cwd", "repo_root"] = "repo_root"
    model: ModelSettings = Field(default_factory=ModelSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    distillation: DistillationSettings = Field(default_factory=DistillationSettings)
    lora: LoraConfig = Field(default_factory=default_lora_config)
    training: SFTConfig = Field(default_factory=default_trainer_config)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    wandb: WandbSettings = Field(default_factory=WandbSettings)

    @field_validator("lora", mode="before")
    @classmethod
    def merge_lora_defaults(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        try:
            return LoraConfig(**{**LORA_DEFAULTS, **value})
        except TypeError as error:
            raise ValueError(str(error)) from error

    @field_validator("training", mode="before")
    @classmethod
    def merge_trainer_defaults(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        try:
            return SFTConfig(**{**TRAINER_DEFAULTS, **value})
        except TypeError as error:
            raise ValueError(str(error)) from error

    @model_validator(mode="after")
    def validate_custom_loss_options(self) -> ExperimentConfig:
        unsupported = [name for name in ("packing", "padding_free", "use_liger_kernel") if getattr(self.training, name)]
        if unsupported:
            raise ValueError("ShiftDistillationTrainer requires these SFTConfig options to be false: " + ", ".join(unsupported))
        return self


@dataclass(frozen=True)
class ModelSpec:
    """Resolved sources needed to instantiate matching base models."""

    weights: str | None
    config: dict[str, Any] | Path | None
    tokenizer: str
    initialization_seed: int


def load_experiment_config(path: str | None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    raw = config_path.read_text()
    if config_path.suffix.lower() == ".json":
        return ExperimentConfig.model_validate(json.loads(raw))
    return ExperimentConfig.model_validate(yaml.safe_load(raw))


def add_training_overrides(parser: Any) -> None:
    """Small experiment-specific CLI surface; SFT options live in YAML."""

    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--vocab-seed", type=int, default=42)
    parser.add_argument("--max-eval-sequences", type=int, default=16)
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "online", "offline"),
        default="disabled",
    )
    parser.add_argument("--wandb-project", default="stego-kirchenbauer-binary-classification")
    parser.add_argument("--wandb-run-name", default="qwen3-4b-base-fineweb-64k")
    parser.add_argument("--wandb-entity")


def find_config_argument(argv: list[str] | None = None) -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    known, _ = parser.parse_known_args(argv)
    return known.config


def _path_base(config: ExperimentConfig) -> Path:
    return REPO_ROOT if config.paths_relative_to == "repo_root" else Path.cwd()


def _resolve_path(value: str, config: ExperimentConfig) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _path_base(config) / path
    return str(path.resolve())


def model_spec_from_config(config: ExperimentConfig) -> ModelSpec:
    settings = config.model
    weights = _resolve_path(settings.weights_path, config) if settings.weights_path is not None else settings.weights
    model_config: dict[str, Any] | Path | None
    if settings.config_path is not None:
        model_config = Path(_resolve_path(settings.config_path, config))
    else:
        model_config = settings.config
    tokenizer = _resolve_path(settings.tokenizer_path, config) if settings.tokenizer_path is not None else settings.tokenizer
    if tokenizer is None:
        raise AssertionError("validated configuration has no tokenizer")
    return ModelSpec(
        weights=weights,
        config=model_config,
        tokenizer=tokenizer,
        initialization_seed=settings.initialization_seed,
    )


def stage_defaults(config: ExperimentConfig, stage: str) -> dict[str, Any]:
    """Flatten nested configuration into existing CLI argument names."""

    paths = artifact_paths()
    defaults = {
        **config.data.model_dump(),
        **config.runtime.model_dump(),
        **config.distillation.model_dump(),
        **{f"wandb_{key}": value for key, value in config.wandb.model_dump().items()},
        "model_spec": model_spec_from_config(config),
        "lora_config": copy.deepcopy(config.lora),
        "trainer_config": copy.deepcopy(config.training),
    }
    if stage == "prefix":
        defaults["output_dir"] = str(paths.prefix_adapter)
    elif stage == "encoding":
        defaults["input_model"] = str(paths.prefix_adapter)
        defaults["output_dir"] = str(paths.encoding_adapter)
    elif stage == "evaluation":
        defaults["input_model"] = str(paths.encoding_adapter)
        defaults["output_dir"] = str(paths.generation_evaluation)
        defaults.update(config.generation.model_dump())
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return defaults


def configured_parser(
    parser: argparse.ArgumentParser,
    *,
    stage: Literal["prefix", "encoding", "evaluation"],
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Apply config defaults, then let explicit CLI flags override them."""

    config_path = find_config_argument(argv)
    config = load_experiment_config(config_path)
    parser.set_defaults(**stage_defaults(config, stage))
    args = parser.parse_args(argv)
    args.experiment_config = config
    return args
