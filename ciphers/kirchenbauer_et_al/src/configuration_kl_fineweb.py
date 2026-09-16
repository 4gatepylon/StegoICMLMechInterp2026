"""Configure Qwen FineWeb training with the gated prefix-KL objective."""

# TODO(hadriano): Migrate this CLI from argparse to Click.
import argparse
import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Literal, Self, Sequence

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_yaml import parse_yaml_raw_as
from transformers import PreTrainedTokenizerBase
from trl import SFTConfig

from ciphers.kirchenbauer_et_al.src.data_kl_fineweb import prefix_token_length

REPO_ROOT = Path(__file__).resolve().parents[3]


class DocumentTokenFilter(BaseModel):
    """Inclusive bounds on FineWeb's stored GPT-2 token count, before truncation.

    ``min_gpt2_document_tokens`` defaults to zero; ``max_gpt2_document_tokens=None``
    disables the upper bound. Shared by cache construction, manifests, loading,
    and training configuration so all entry points enforce the same range.
    """

    model_config = ConfigDict(extra="forbid")

    # FineWeb already stores GPT-2 counts in token_count. They are close to Qwen
    # counts but not identical, so this cheap prefilter cannot replace Qwen filtering.
    min_gpt2_document_tokens: int = Field(default=0, ge=0)
    max_gpt2_document_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_token_range(self) -> Self:
        """Return the validated configuration, rejecting reversed bounds."""
        if self.max_gpt2_document_tokens is not None and self.max_gpt2_document_tokens < self.min_gpt2_document_tokens:
            raise ValueError("max_gpt2_document_tokens must be greater than or equal to min_gpt2_document_tokens")
        return self

    def accepts(self, token_count: int) -> bool:
        """Return whether a stored GPT-2 ``token_count`` satisfies both bounds."""
        return token_count >= self.min_gpt2_document_tokens and (self.max_gpt2_document_tokens is None or token_count <= self.max_gpt2_document_tokens)


class QwenDocumentTokenFilter(DocumentTokenFilter):
    """Nested GPT-2 and training-tokenizer bounds on complete document text.

    Inherited bounds use cached GPT-2 counts and are applied first. Qwen bounds
    use the caller's training tokenizer without prefixes, special tokens,
    padding, or truncation. Zero minimum and an unlimited maximum disable the
    Qwen stage, so existing cache consumers need no tokenizer.
    """

    min_qwen_document_tokens: int = Field(default=0, ge=0)
    max_qwen_document_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_qwen_token_range(self) -> Self:
        """Reject reversed Qwen bounds independently of the GPT-2 bounds."""
        if self.max_qwen_document_tokens is not None and self.max_qwen_document_tokens < self.min_qwen_document_tokens:
            raise ValueError("max_qwen_document_tokens must be greater than or equal to min_qwen_document_tokens")
        return self

    def accepts_qwen(self, token_count: int) -> bool:
        """Return whether a complete document's Qwen count meets inclusive bounds."""
        return token_count >= self.min_qwen_document_tokens and (self.max_qwen_document_tokens is None or token_count <= self.max_qwen_document_tokens)


class PrefixKLTrainingConfig(QwenDocumentTokenFilter):
    """Validated settings accepted by the FineWeb KL training entry point."""

    model_config = ConfigDict(extra="forbid")

    # --- PrefixKLTrainer constructor and model setup ---
    model: str = "Qwen/Qwen3-4B-Base"
    loss_mode: Literal["nll", "ignore_prefix"] = "nll"
    strategy: Literal["block", "modulo"] = "block"
    n_bits: int = Field(default=8, gt=0)
    alpha: float = 1.0
    delta: float = 1.0
    profile_memory_steps: int = Field(default=0, ge=0)

    # --- Dataset and collator; SFTConfig adds the tokenized prefix to data_length ---
    dataset_cache_name: str = "fineweb-500k"
    concatenation_space: Literal["token", "character"] = "token"
    data_length: int = Field(default=4096, gt=0)
    validation_samples: int = Field(default=256, gt=0)

    # --- PEFT LoraConfig ---
    lora_rank: int = Field(default=32, gt=0)
    lora_alpha: int = Field(default=16, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)

    # --- TRL SFTConfig ---
    run_name: str = "qwen3-4b-fineweb-prefix-kl-lora"
    max_steps: int = Field(default=10_000, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0)
    warmup_steps: int = Field(default=300, ge=0)
    per_device_batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int | None = Field(default=None, gt=0)
    eval_steps: int = Field(default=4, gt=0)
    save_steps: int = Field(default=500, gt=0)
    save_total_limit: int = Field(default=5, gt=0)
    logging_steps: int = Field(default=1, gt=0)
    include_num_input_tokens_seen: Literal["all", "non_padding", "no"] = "non_padding"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    report_to: str = "wandb"

    # --- Entrypoint orchestration around Trainer ---
    global_batch_size: int | None = Field(default=None, gt=0)
    wandb_project: str | None = None
    wandb_tags: list[str] = Field(default_factory=list)
    resume_from_checkpoint: str | None = None

    @model_validator(mode="after")
    def validate_data_partition(self) -> Self:
        """Reject data budgets that cannot be split evenly across message bits."""
        if self.data_length % self.n_bits:
            raise ValueError("data_length must be divisible by n_bits")
        return self


def build_sft_config(args: PrefixKLTrainingConfig, grad_accumulation_steps: int, tokenizer: PreTrainedTokenizerBase) -> SFTConfig:
    """Translate validated experiment settings into Transformers training arguments.

    Args:
        args: Complete ``PrefixKLTrainingConfig`` produced by ``parse_args()``.
            Its output path is resolved below ``STEGO_ARTIFACTS_DIR``; all
            optimization, checkpoint, precision, and reporting fields are
            forwarded to their ``SFTConfig`` consumers.
        grad_accumulation_steps: Positive per-process accumulation count after
            resolving the configured effective global batch size.
        tokenizer: The training collator's tokenizer, used to measure the
            fixed-width control prefix without added special tokens.

    Returns:
        The ``SFTConfig`` consumed by ``PrefixKLTrainer``. In particular,
        ``include_num_input_tokens_seen`` controls Transformers' cumulative
        all-token or non-padding-token counter, while ``PrefixKLTrainer.log()``
        independently adds the cumulative padded-token counter. ``max_length``
        includes both the data budget and measured prefix width. Rank zero
        prints this relationship before training starts; each collated batch
        checks that its prefixes match the reference width.
    """
    prefix_length = prefix_token_length(tokenizer, args.n_bits)
    max_length = args.data_length + prefix_length
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"[prefix-KL] data_length={args.data_length} + prefix_length={prefix_length} = max_length={max_length} tokens per model input. "
            f"The control prefix contains a {args.n_bits}-bit secret message and the encoding gate; "
            f"the {args.strategy} partition assigns {args.data_length // args.n_bits} data positions per bit. "
            "Short documents are padded to data_length; total-input token metrics include the prefix.",
            flush=True,
        )
    return SFTConfig(
        output_dir=os.path.join(os.environ["STEGO_ARTIFACTS_DIR"], args.run_name),
        run_name=args.run_name,
        report_to=args.report_to,
        max_length=max_length,
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
        include_num_input_tokens_seen=args.include_num_input_tokens_seen,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        loss_type="nll",
        model_init_kwargs={"torch_dtype": getattr(torch, args.dtype)},
    )


def load_training_config(config_path: str | None) -> PrefixKLTrainingConfig:
    """Load validated training defaults from a repository-relative YAML file.

    Args:
        config_path: Path relative to the repository root. ``None`` selects the
            historical command-line defaults instead of reading a file.

    Returns:
        A fully populated ``PrefixKLTrainingConfig``. ``parse_args`` uses its
        fields as parser defaults, so explicitly supplied command-line flags take
        precedence.
    """
    if config_path is None:
        return PrefixKLTrainingConfig()
    relative_config_path = Path(config_path)
    if relative_config_path.is_absolute():
        raise ValueError("config path must be relative to the repository root")
    return parse_yaml_raw_as(PrefixKLTrainingConfig, (REPO_ROOT / relative_config_path).read_text())


def parse_args(argv: Sequence[str] | None = None) -> PrefixKLTrainingConfig:
    """Parse and validate YAML-backed command-line training settings.

    Args:
        argv: Arguments without the program name. ``None`` reads ``sys.argv``.

    Returns:
        A ``PrefixKLTrainingConfig`` containing every trainer, objective, data,
        batching, checkpointing, precision, and reporting field consumed by the
        training entry point. Explicit command-line arguments override values
        loaded through ``--config``.
    """
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", help="repository-relative YAML configuration file")
    config_args, _ = config_parser.parse_known_args(argv)
    config = load_training_config(config_args.config)

    parser = argparse.ArgumentParser(description="LoRA-train Qwen on FineWeb with the configurable gated prefix KL objective.")
    add = parser.add_argument
    add("--config", help="repository-relative YAML configuration file")
    add("--model", default="Qwen/Qwen3-4B-Base")
    add("--run-name", default="qwen3-4b-fineweb-prefix-kl-lora")
    add(
        "--dataset-cache-name",
        default="fineweb-500k",
        help="completed cache below $STEGO_ARTIFACTS_DIR/datasets/fineweb (default: fineweb-500k)",
    )
    add("--loss-mode", choices=("nll", "ignore_prefix"), default="nll")
    add("--min-gpt2-document-tokens", type=int, help="inclusive minimum stored GPT-2 token count (default: 0)")
    add("--max-gpt2-document-tokens", type=lambda value: None if value.lower() == "none" else int(value), help="inclusive maximum stored GPT-2 token count (default: none)")
    add("--min-qwen-document-tokens", type=int, help="inclusive minimum full-document training-tokenizer count, after GPT-2 filtering (default: 0)")
    add(
        "--max-qwen-document-tokens",
        type=lambda value: None if value.lower() == "none" else int(value),
        help="inclusive maximum full-document training-tokenizer count, after GPT-2 filtering (default: none)",
    )
    add("--strategy", choices=("block", "modulo"), default="block")
    add("--concatenation-space", choices=("token", "character"), default="token")
    add("--n-bits", type=int, default=8)
    add("--alpha", type=float, default=1.0)
    add("--delta", type=float, default=1.0)
    add("--data-length", type=int, default=4096, help="document token slots excluding the control prefix; must be divisible by n-bits (default: 4096)")
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
    add("--validation-samples", type=int, default=256)
    add("--eval-steps", type=int, default=4)
    add("--save-steps", type=int, default=500)
    add("--save-total-limit", type=int, default=5)
    add("--logging-steps", type=int, default=1)
    add(
        "--include-num-input-tokens-seen",
        choices=("all", "non_padding", "no"),
        default="non_padding",
        help="Track all, non-padding, or no prefixed input tokens in Trainer logs (default: non_padding)",
    )
    add("--lora-rank", type=int, default=32)
    add("--lora-alpha", type=int, default=16)
    add("--lora-dropout", type=float, default=0.05)
    add("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    add("--report-to", default="wandb")
    add("--wandb-project")
    add("--wandb-tag", action="append", dest="wandb_tags")
    add("--resume-from-checkpoint")
    add("--profile-memory-steps", type=int, default=0, help="Profile this many initial microbatches per rank")
    parser.set_defaults(**config.model_dump())
    parsed_arguments = vars(parser.parse_args(argv))
    parsed_arguments.pop("config")
    return PrefixKLTrainingConfig.model_validate(parsed_arguments)


def configure_wandb_environment(args: PrefixKLTrainingConfig, environment: MutableMapping[str, str]) -> None:
    """Expose validated reporting settings to the W&B initialization consumer.

    Transformers creates the W&B run after the training entry point constructs
    its trainer. The W&B SDK consumes ``WANDB_PROJECT`` and the comma-separated
    ``WANDB_TAGS`` setting at that later initialization boundary, so this helper
    applies the validated configuration before trainer construction.

    Args:
        args: Complete training configuration. ``wandb_project`` optionally
            selects the destination project, while ``wandb_tags`` adds tags that
            identify official runs. An empty tag list does not mark ordinary or
            dummy runs.
        environment: Mutable process environment consumed by the W&B SDK.
            Existing ``WANDB_TAGS`` values are preserved in their original order;
            configured tags are appended once.

    Returns:
        ``None``. Callers use the mutated environment when Transformers later
        initializes W&B. This function does not contact W&B or create a run.
    """
    if args.wandb_project is not None:
        environment["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_tags:
        existing_tags = [tag for tag in environment.get("WANDB_TAGS", "").split(",") if tag]
        environment["WANDB_TAGS"] = ",".join(dict.fromkeys([*existing_tags, *args.wandb_tags]))


def gradient_accumulation_steps(args: PrefixKLTrainingConfig, world_size: int) -> int:
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
