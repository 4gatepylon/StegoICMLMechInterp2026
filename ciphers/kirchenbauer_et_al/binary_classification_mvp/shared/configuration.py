"""JSON/YAML experiment configuration and CLI-default resolution."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constants import MODEL_NAME, REPO_ROOT


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


class TrainingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delta: float = 2.0
    vocab_seed: int = 42
    training_seed: int = 1234
    epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    logit_chunk_size: int = 32
    log_every_steps: int = 8
    eval_every_steps: int = 32
    max_eval_sequences: int = 16
    max_steps: int | None = None


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str = "auto"
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"


class GenerationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sampling_seed: int = 2026
    temperature: float = 0.7
    max_new_tokens: int = 200


class OutputSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prefix: str = "ciphers/kirchenbauer_et_al/binary_classification_mvp/outputs/prefix_adapter"
    encoding: str = "ciphers/kirchenbauer_et_al/binary_classification_mvp/outputs/encoding_adapter"
    evaluation: str = "ciphers/kirchenbauer_et_al/binary_classification_mvp/outputs/generation_evaluation"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths_relative_to: Literal["cwd", "repo_root"] = "repo_root"
    model: ModelSettings = Field(default_factory=ModelSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    outputs: OutputSettings = Field(default_factory=OutputSettings)


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

    defaults = {
        **config.data.model_dump(),
        **config.runtime.model_dump(),
        **config.training.model_dump(),
        "model_spec": model_spec_from_config(config),
    }
    if stage == "prefix":
        defaults["output_dir"] = _resolve_path(config.outputs.prefix, config)
    elif stage == "encoding":
        defaults["input_model"] = _resolve_path(config.outputs.prefix, config)
        defaults["output_dir"] = _resolve_path(config.outputs.encoding, config)
    elif stage == "evaluation":
        defaults["input_model"] = _resolve_path(config.outputs.encoding, config)
        defaults["output_dir"] = _resolve_path(config.outputs.evaluation, config)
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
