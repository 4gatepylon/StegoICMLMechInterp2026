"""Ablate Qwen3 base-model size and the prefix-KL objective with compact LoRA saves."""

import os
from functools import partial
from pathlib import Path
from typing import Literal, Self, get_args

import click
from peft import LoraConfig
from pydantic import Field, model_validator
from transformers import AutoTokenizer

from ciphers.kirchenbauer_et_al.src.cache_fineweb import load_fineweb_cache
from ciphers.kirchenbauer_et_al.src.configuration_kl_fineweb import (
    PrefixKLTrainingConfig,
    build_sft_config,
    configure_wandb_environment,
    gradient_accumulation_steps,
)
from ciphers.kirchenbauer_et_al.src.data_kl_fineweb import fixed_prefix_metadata
from ciphers.kirchenbauer_et_al.src.trainer_kl_fineweb import PrefixKLTrainer, prefix_bits_encoding_text_collator

QwenModel = Literal["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base"]
LossType = Literal["nll", "ignore_prefix"]
WANDB_PROJECT = "E20260916_qwen3_peft_convergence"
SEQUENCE_LENGTH = 4096
# Original run: 128 sequences per optimizer step, 1,024 steps, padded to this length.
NUM_TRAINING_TOKENS = 128 * 1024 * SEQUENCE_LENGTH


class FixedBudgetTrainingConfig(PrefixKLTrainingConfig):
    """Derive training steps from a budget of padded token positions."""

    model: QwenModel = "Qwen/Qwen3-0.6B-Base"
    learning_rate: float = Field(default=3e-4, gt=0, allow_inf_nan=False)
    alpha: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    delta: float = Field(default=2.0, ge=0, allow_inf_nan=False)
    num_training_tokens: int = Field(default=NUM_TRAINING_TOKENS, gt=0)

    @model_validator(mode="after")
    def derive_run_identity(self) -> Self:
        """Return settings with a shared project and a descriptive run/output name.

        Model size, bits, learning rate, global batch, loss, alpha, and delta
        distinguish ablations. Repeats with identical settings reuse the name;
        use a fresh STEGO_ARTIFACTS_DIR for independent checkpoint outputs.
        The explicit project overrides WANDB_PROJECT through build_trainer.
        """
        model_name = self.model.split("/")[1].removesuffix("-Base").lower()
        self.wandb_project = WANDB_PROJECT
        self.run_name = f"{model_name}-{self.n_bits}bit-lr{self.learning_rate:g}-gb{self.global_batch_size}-{self.loss_mode}-a{self.alpha:g}-d{self.delta:g}"
        return self

    @model_validator(mode="after")
    def derive_step_budget(self) -> Self:
        """Return settings with exact-budget steps and retention for every save.

        Reject batches whose token count cannot divide the training token budget;
        rounding would change the number of padded training tokens processed.
        Retention includes the Trainer's final save for a partial save interval.
        """
        if self.global_batch_size is None or self.num_training_tokens % (self.max_length * self.global_batch_size):
            raise ValueError(
                f"num_training_tokens ({self.num_training_tokens}) must be divisible by sequence length ({self.max_length}) * global batch size ({self.global_batch_size})"
            )
        self.max_steps = self.num_training_tokens // self.max_length // self.global_batch_size
        self.save_total_limit = (self.max_steps + self.save_steps - 1) // self.save_steps
        return self


def experiment_config(
    local_batch_size: int = 8,
    global_batch_size: int = 128,
    num_training_tokens: int = NUM_TRAINING_TOKENS,
    *,
    lr: float = 3e-4,
    model: QwenModel = "Qwen/Qwen3-0.6B-Base",
    n_bits: int = 1,
    loss_type: LossType = "nll",
    alpha: float = 1.0,
    delta: float = 2.0,
) -> FixedBudgetTrainingConfig:
    """Return validated ablation settings consumed by ``build_trainer``.

    ``local_batch_size`` is sequences per device per microbatch; ``global_batch_size``
    includes all devices and accumulation. ``num_training_tokens`` counts padded
    training positions, excluding validation; it must divide exactly into steps.
    ``lr`` controls optimizer update size. ``model`` selects a supported Qwen3 Base
    checkpoint; ``n_bits`` sets message length and the number of text blocks.
    ``loss_type='nll'`` minimizes prefix NLL plus ``alpha`` times data KL;
    ``'ignore_prefix'`` uses only data KL and ignores alpha. ``delta`` boosts the
    teacher's selected vocabulary logits: zero disables the boost; larger values
    strengthen the encoding target. Alpha and delta must be finite and nonnegative.
    The returned config also derives W&B project/run names and checkpoint retention.
    """
    return FixedBudgetTrainingConfig(
        num_training_tokens=num_training_tokens,
        model=model,
        loss_mode=loss_type,
        strategy="block",
        n_bits=n_bits,
        alpha=alpha,
        delta=delta,
        dataset_cache_name="fineweb-500k",
        concatenation_space="token",
        max_length=SEQUENCE_LENGTH,
        validation_samples=256,
        lora_rank=32,
        lora_alpha=16,
        lora_dropout=0.05,
        learning_rate=lr,
        warmup_steps=50,
        global_batch_size=global_batch_size,
        per_device_batch_size=local_batch_size,
        eval_steps=4,
        save_steps=32,
        logging_steps=1,
        report_to="wandb",
        wandb_tags=["stego-icml-2026-git-archive"],
    )


def build_trainer(config: PrefixKLTrainingConfig) -> PrefixKLTrainer:
    """Build the shared KL trainer with experiment-specific compact saves.

    Args:
        config: Validated model, data, objective, batching, and logging settings,
            normally from ``experiment_config``. ``WORLD_SIZE`` determines
            accumulation; ``STEGO_ARTIFACTS_DIR`` locates data and outputs.
            ``config.wandb_project`` selects the destination W&B project.

    Returns:
        An untrained ``PrefixKLTrainer`` with a PEFT LoRA model. Call ``train``
        to optimize it. Checkpoints contain adapters and tokenizer/Trainer
        metadata, but no base weights, optimizer, scheduler, or RNG state;
        they are for evaluation, not exact training resumption. The shared
        collator and fixed validation metadata must stay paired with this
        trainer so teacher/student token positions and validation gates align.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    accumulation_steps = gradient_accumulation_steps(config, world_size)
    training_args = build_sft_config(config, accumulation_steps)
    training_args.save_only_model = True
    configure_wandb_environment(config, os.environ)
    # W&B otherwise writes its local logs relative to the working directory.
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = training_args.output_dir
    required_documents = config.validation_samples + config.max_steps * config.per_device_batch_size * world_size * accumulation_steps
    dataset = load_fineweb_cache(
        config.dataset_cache_name,
        minimum_documents=required_documents,
        min_document_tokens=config.min_document_tokens,
        max_document_tokens=config.max_document_tokens,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    validation_dataset = dataset.take(config.validation_samples).map(fixed_prefix_metadata, with_indices=True, fn_kwargs={"n_bits": config.n_bits})
    return PrefixKLTrainer(
        model=config.model,
        loss_mode=config.loss_mode,
        alpha=config.alpha,
        n_bits=config.n_bits,
        delta=config.delta,
        strategy=config.strategy,
        train_dataset=dataset.skip(config.validation_samples),
        eval_dataset=validation_dataset,
        data_collator=partial(
            prefix_bits_encoding_text_collator,
            tokenizer=tokenizer,
            n_bits=config.n_bits,
            max_length=config.max_length,
            concatenation_space=config.concatenation_space,
        ),
        processing_class=tokenizer,
        peft_config=LoraConfig(task_type="CAUSAL_LM", r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules="all-linear"),
        args=training_args,
    )


@click.command()
@click.option(
    "--lr",
    default=3e-4,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
    help="Optimizer learning rate. Increase for faster updates; decrease if training is unstable.",
)
@click.option(
    "--model",
    "--model-name",
    default="Qwen/Qwen3-0.6B-Base",
    show_default=True,
    type=click.Choice(get_args(QwenModel)),
    help="Qwen3 Base checkpoint. Larger models test capacity scaling but require more memory and compute.",
)
@click.option(
    "--n-bits",
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help="Message bits per sequence. More bits increase payload and divide the text into shorter blocks per bit; use 1, 2, 4, or 8 for this sweep.",
)
@click.option(
    "--loss-type",
    default="nll",
    show_default=True,
    type=click.Choice(get_args(LossType)),
    help="nll: prefix NLL + alpha * data KL. ignore_prefix: data KL only (alpha ignored), to ablate prefix learning.",
)
@click.option(
    "--alpha",
    default=1.0,
    show_default=True,
    type=click.FloatRange(min=0),
    help="Data-KL weight in nll mode. Increase to prioritize matching the encoding target over prefix NLL; 0 trains only the prefix. Ignored by ignore_prefix.",
)
@click.option(
    "--delta",
    default=2.0,
    show_default=True,
    type=click.FloatRange(min=0),
    help="Logit boost for the vocabulary subset encoding each bit. Larger values strengthen the encoding target and may trade text quality for bit recovery; 0 disables the boost.",
)
@click.option(
    "--local-batch-size",
    default=8,
    show_default=True,
    type=click.IntRange(min=1),
    help="Sequences per device per microbatch. Lower to reduce memory use; accumulation preserves the global batch.",
)
@click.option(
    "--global-batch-size",
    default=128,
    show_default=True,
    type=click.IntRange(min=1),
    help="Sequences per optimizer step across all devices. Lower for more updates at the same token budget; must be divisible by local batch * WORLD_SIZE.",
)
@click.option(
    "--num-training-tokens",
    default=NUM_TRAINING_TOKENS,
    show_default=True,
    type=click.IntRange(min=1),
    help="Total padded training token positions, excluding validation. Increase to train longer at a fixed batch size.",
)
def main(
    local_batch_size: int, global_batch_size: int, num_training_tokens: int, lr: float, model: QwenModel, n_bits: int, loss_type: LossType, alpha: float, delta: float
) -> None:
    """Train for a token budget, deriving optimizer steps from sequence and batch sizes.

    With the default budget, --local-batch-size 4 --global-batch-size 32 runs 4,096 steps.
    Sequence length * global batch must divide the token budget exactly.
    Global batch must be divisible by local batch * WORLD_SIZE.
    Checkpoints remain every 32 optimizer steps; all are retained.
    """
    try:
        trainer = build_trainer(
            experiment_config(local_batch_size, global_batch_size, num_training_tokens, lr=lr, model=model, n_bits=n_bits, loss_type=loss_type, alpha=alpha, delta=delta)
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    trainer.train()


if __name__ == "__main__":
    main()
