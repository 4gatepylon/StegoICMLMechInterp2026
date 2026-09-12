"""LoRA-train Qwen on FineWeb with the configurable gated prefix KL objective."""

# TODO(hadriano): Migrate this CLI from argparse to Click.
import argparse
import os
import sys
from functools import partial
from pathlib import Path
from typing import Literal, Sequence

import torch
from peft import LoraConfig
from pydantic import BaseModel, ConfigDict, Field
from pydantic_yaml import parse_yaml_raw_as
from transformers import AutoTokenizer
from trl import SFTConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ciphers.kirchenbauer_et_al.binary_classification_mvp.data import fixed_prefix_metadata, load_fineweb  # noqa: E402
from ciphers.kirchenbauer_et_al.binary_classification_mvp.kl_trainer import PrefixKLTrainer, prefix_bits_encoding_text_collator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]


class TrainingConfig(BaseModel):
    """Validated settings accepted by the FineWeb KL training entry point."""

    model_config = ConfigDict(extra="forbid")

    model: str = "Qwen/Qwen3-4B-Base"
    run_name: str = "qwen3-4b-fineweb-prefix-kl-lora"
    loss_mode: Literal["nll", "ignore_prefix"] = "nll"
    strategy: Literal["block", "modulo"] = "block"
    concatenation_space: Literal["token", "character"] = "token"
    n_bits: int = Field(default=8, gt=0)
    alpha: float = 1.0
    delta: float = 1.0
    max_length: int = Field(default=4096, gt=0)
    max_steps: int = Field(default=10_000, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0)
    warmup_steps: int = Field(default=300, ge=0)
    global_batch_size: int | None = Field(default=None, gt=0)
    per_device_batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int | None = Field(default=None, gt=0)
    validation_samples: int = Field(default=1_000, gt=0)
    eval_steps: int = Field(default=4, gt=0)
    save_steps: int = Field(default=500, gt=0)
    save_total_limit: int = Field(default=2, gt=0)
    logging_steps: int = Field(default=1, gt=0)
    lora_rank: int = Field(default=32, gt=0)
    lora_alpha: int = Field(default=16, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    report_to: str = "wandb"
    wandb_project: str | None = None
    resume_from_checkpoint: str | None = None
    profile_memory_steps: int = Field(default=0, ge=0)


def load_training_config(config_path: str | None) -> TrainingConfig:
    """Load validated training defaults from a repository-relative YAML file.

    Args:
        config_path: Path relative to the repository root. ``None`` selects the
            historical command-line defaults instead of reading a file.

    Returns:
        A fully populated ``TrainingConfig``. ``parse_args`` uses its fields as
        parser defaults, so explicitly supplied command-line flags take precedence.
    """
    if config_path is None:
        return TrainingConfig()
    relative_config_path = Path(config_path)
    if relative_config_path.is_absolute():
        raise ValueError("config path must be relative to the repository root")
    return parse_yaml_raw_as(TrainingConfig, (REPO_ROOT / relative_config_path).read_text())


def parse_args(argv: Sequence[str] | None = None) -> TrainingConfig:
    """Parse and validate YAML-backed command-line training settings.

    Args:
        argv: Arguments without the program name. ``None`` reads ``sys.argv``.

    Returns:
        A ``TrainingConfig`` containing every trainer, objective, data, batching,
        checkpointing, precision, and reporting field consumed by ``main``.
        Explicit command-line arguments override values loaded through ``--config``.
    """
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", help="repository-relative YAML configuration file")
    config_args, _ = config_parser.parse_known_args(argv)
    config = load_training_config(config_args.config)

    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--config", help="repository-relative YAML configuration file")
    add("--model", default="Qwen/Qwen3-4B-Base")
    add("--run-name", default="qwen3-4b-fineweb-prefix-kl-lora")
    add("--loss-mode", choices=("nll", "ignore_prefix"), default="nll")
    add("--strategy", choices=("block", "modulo"), default="block")
    add("--concatenation-space", choices=("token", "character"), default="token")
    add("--n-bits", type=int, default=8)
    add("--alpha", type=float, default=1.0)
    add("--delta", type=float, default=1.0)
    add("--max-length", type=int, default=4096)
    add("--max-steps", type=int, default=10_000)
    add("--learning-rate", "--lr", type=float, default=3e-4)
    add("--warmup-steps", type=int, default=300)
    add(
        "--global-batch-size",
        type=int,
        help="derive gradient accumulation for this effective batch size (default: 32)",
    )
    add("--per-device-batch-size", "--batch-size", type=int, default=1)
    add("--gradient-accumulation-steps", "--grad-accum-steps", type=int)
    add("--validation-samples", type=int, default=1_000)
    add("--eval-steps", type=int, default=4)
    add("--save-steps", type=int, default=500)
    add("--save-total-limit", type=int, default=2)
    add("--logging-steps", type=int, default=1)
    add("--lora-rank", type=int, default=32)
    add("--lora-alpha", type=int, default=16)
    add("--lora-dropout", type=float, default=0.05)
    add("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    add("--report-to", default="wandb")
    add("--wandb-project")
    add("--resume-from-checkpoint")
    add("--profile-memory-steps", type=int, default=0, help="Profile this many initial microbatches per rank")
    parser.set_defaults(**config.model_dump())
    parsed_arguments = vars(parser.parse_args(argv))
    parsed_arguments.pop("config")
    return TrainingConfig.model_validate(parsed_arguments)


def gradient_accumulation_steps(args: TrainingConfig, world_size: int) -> int:
    if args.per_device_batch_size <= 0:
        raise ValueError("per-device batch size must be positive")
    if args.gradient_accumulation_steps is not None:
        if args.global_batch_size is not None:
            raise ValueError("set either global batch size or gradient accumulation steps, not both")
        if args.gradient_accumulation_steps <= 0:
            raise ValueError("gradient accumulation steps must be positive")
        return args.gradient_accumulation_steps

    global_batch_size = args.global_batch_size if args.global_batch_size is not None else 32
    micro_batch = args.per_device_batch_size * world_size
    if global_batch_size <= 0:
        raise ValueError("global batch size must be positive")
    if global_batch_size % micro_batch:
        raise ValueError("global batch size must be divisible by per-device batch size * WORLD_SIZE")
    return global_batch_size // micro_batch


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    grad_accumulation_steps = gradient_accumulation_steps(args, world_size)
    if args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    dataset = load_fineweb()
    validation_dataset = dataset.take(args.validation_samples).map(
        fixed_prefix_metadata,
        with_indices=True,
        fn_kwargs={"n_bits": args.n_bits},
    )
    trainer = PrefixKLTrainer(
        model=args.model,
        loss_mode=args.loss_mode,
        alpha=args.alpha,
        n_bits=args.n_bits,
        delta=args.delta,
        strategy=args.strategy,
        profile_memory_steps=args.profile_memory_steps,
        train_dataset=dataset.skip(args.validation_samples),
        eval_dataset=validation_dataset,
        data_collator=partial(
            prefix_bits_encoding_text_collator, tokenizer=tokenizer, n_bits=args.n_bits, max_length=args.max_length, concatenation_space=args.concatenation_space
        ),
        processing_class=tokenizer,
        peft_config=LoraConfig(task_type="CAUSAL_LM", r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules="all-linear"),
        args=SFTConfig(
            output_dir=os.path.join(os.environ["STEGO_ARTIFACTS_DIR"], args.run_name),
            run_name=args.run_name,
            report_to=args.report_to,
            max_length=args.max_length,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=grad_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            bf16=args.dtype == "bfloat16",
            fp16=args.dtype == "float16",
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False,
            logging_steps=args.logging_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            model_init_kwargs={"torch_dtype": getattr(torch, args.dtype)},
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
