"""Test one-bit prefix-KL convergence of Qwen3-0.6B-Base with compact LoRA saves."""

import os
from functools import partial
from pathlib import Path
from typing import Self

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

SEQUENCE_LENGTH = 4096
# Preserve the original schedule's token counts at global batch 128.
REFERENCE_TOKENS_PER_STEP = 128 * SEQUENCE_LENGTH
NUM_TRAINING_TOKENS = 1024 * REFERENCE_TOKENS_PER_STEP


class FixedBudgetTrainingConfig(PrefixKLTrainingConfig):
    """Express this experiment's schedule in padded student-input token positions.

    num_training_tokens excludes validation and the separate teacher forward pass.
    The *_steps_in_tokens fields specify warmup duration and event intervals;
    their defaults preserve the original batch-128, length-4096 schedule. The
    inherited step fields and checkpoint retention are derived on validation for
    build_sft_config; configure the token fields rather than those derived fields.
    """

    num_training_tokens: int = Field(default=NUM_TRAINING_TOKENS, gt=0)
    warmup_steps_in_tokens: int = Field(default=50 * REFERENCE_TOKENS_PER_STEP, ge=0)
    eval_steps_in_tokens: int = Field(default=4 * REFERENCE_TOKENS_PER_STEP, gt=0)
    save_steps_in_tokens: int = Field(default=32 * REFERENCE_TOKENS_PER_STEP, gt=0)
    logging_steps_in_tokens: int = Field(default=REFERENCE_TOKENS_PER_STEP, gt=0)

    @model_validator(mode="after")
    def derive_step_budget(self) -> Self:
        """Return exact-budget training steps, token-based intervals and retention.

        Reject batches whose token count cannot divide the training token budget;
        rounding would change the number of padded training tokens processed.
        Round event intervals and warmup up to whole optimizer steps, so each is
        at least its requested token count; zero warmup remains zero. Interval
        rounding can overshoot by less than one step per event interval.
        Retention includes the Trainer's final save for a partial save interval.
        """
        if self.global_batch_size is None or self.num_training_tokens % (self.max_length * self.global_batch_size):
            raise ValueError(
                f"num_training_tokens ({self.num_training_tokens}) must be divisible by sequence length ({self.max_length}) * global batch size ({self.global_batch_size})"
            )
        tokens_per_step = self.max_length * self.global_batch_size
        self.max_steps = self.num_training_tokens // tokens_per_step
        self.warmup_steps = (self.warmup_steps_in_tokens + tokens_per_step - 1) // tokens_per_step
        self.eval_steps = (self.eval_steps_in_tokens + tokens_per_step - 1) // tokens_per_step
        self.save_steps = (self.save_steps_in_tokens + tokens_per_step - 1) // tokens_per_step
        self.logging_steps = (self.logging_steps_in_tokens + tokens_per_step - 1) // tokens_per_step
        self.save_total_limit = (self.max_steps + self.save_steps - 1) // self.save_steps
        return self


def experiment_config(local_batch_size: int = 8, global_batch_size: int = 128, num_training_tokens: int = NUM_TRAINING_TOKENS) -> FixedBudgetTrainingConfig:
    """Return validated settings for the one-bit 0.6B convergence experiment.

    The returned schema is consumed by ``build_trainer``. Settings follow the
    existing one-bit 4B experiment, changing the model, local batch, run name,
    and checkpoint cadence/retention. ``local_batch_size`` is the per-device
    microbatch; ``global_batch_size`` is the effective batch across devices and
    accumulation. ``num_training_tokens`` is the positive training budget in
    padded token positions, excluding validation. Steps divide this budget by
    ``SEQUENCE_LENGTH`` and the global batch size; a non-integer result is rejected.
    Warmup, evaluation, saving and logging preserve their original token intervals
    using the local schema's *_steps_in_tokens defaults, rounded up to full steps.
    """
    return FixedBudgetTrainingConfig(
        num_training_tokens=num_training_tokens,
        model="Qwen/Qwen3-0.6B-Base",
        run_name="E20260916_qwen3_0_6b_peft_convergence",
        loss_mode="nll",
        strategy="block",
        n_bits=1,
        alpha=1.0,
        delta=2.0,
        dataset_cache_name="fineweb-500k",
        concatenation_space="token",
        max_length=SEQUENCE_LENGTH,
        validation_samples=256,
        lora_rank=32,
        lora_alpha=16,
        lora_dropout=0.05,
        learning_rate=3e-4,
        global_batch_size=global_batch_size,
        per_device_batch_size=local_batch_size,
        report_to="wandb",
        wandb_tags=["stego-icml-2026-git-archive"],
    )


def build_trainer(config: PrefixKLTrainingConfig) -> PrefixKLTrainer:
    """Build the shared KL trainer with experiment-specific compact saves.

    Args:
        config: Validated model, data, objective, batching, and logging settings,
            normally from ``experiment_config``. ``WORLD_SIZE`` determines
            accumulation; ``STEGO_ARTIFACTS_DIR`` locates data and outputs.
            Existing ``WANDB_PROJECT`` selects the destination project.

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
@click.option("--local-batch-size", default=8, show_default=True, type=click.IntRange(min=1), help="Sequences per device per microbatch.")
@click.option("--global-batch-size", default=128, show_default=True, type=click.IntRange(min=1), help="Sequences per optimizer step across all devices.")
@click.option(
    "--num-training-tokens", default=NUM_TRAINING_TOKENS, show_default=True, type=click.IntRange(min=1), help="Total padded training token positions; excludes validation."
)
def main(local_batch_size: int, global_batch_size: int, num_training_tokens: int) -> None:
    """Train for a token budget, deriving optimizer steps from sequence and batch sizes.

    With the default budget, --local-batch-size 4 --global-batch-size 32 runs 4,096 steps.
    Sequence length * global batch must divide the token budget exactly.
    Global batch must be divisible by local batch * WORLD_SIZE.
    At global batch 32, evaluation is every 16 steps and saving every 128 steps;
    their padded-token intervals match the original batch-128 run. All saves are retained.
    """
    try:
        trainer = build_trainer(experiment_config(local_batch_size, global_batch_size, num_training_tokens))
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    trainer.train()


if __name__ == "__main__":
    main()
