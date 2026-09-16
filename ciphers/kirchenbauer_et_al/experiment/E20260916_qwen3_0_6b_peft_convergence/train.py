"""Test one-bit prefix-KL convergence of Qwen3-0.6B-Base with compact LoRA saves."""

import os
from functools import partial
from pathlib import Path

import click
from peft import LoraConfig
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


def experiment_config() -> PrefixKLTrainingConfig:
    """Return validated settings for the one-bit 0.6B convergence experiment.

    The returned schema is consumed by ``build_trainer``. Settings follow the
    existing one-bit 4B experiment, changing the model, local batch, run name,
    and checkpoint cadence/retention. Steps count optimizer updates, so the
    effective batch stays 128 regardless of the supported process count.
    """
    return PrefixKLTrainingConfig(
        model="Qwen/Qwen3-0.6B-Base",
        run_name="E20260916_qwen3_0_6b_peft_convergence",
        loss_mode="nll",
        strategy="block",
        n_bits=1,
        alpha=1.0,
        delta=2.0,
        dataset_cache_name="fineweb-500k",
        concatenation_space="token",
        max_length=4096,
        validation_samples=256,
        lora_rank=32,
        lora_alpha=16,
        lora_dropout=0.05,
        max_steps=1024,
        learning_rate=3e-4,
        warmup_steps=50,
        global_batch_size=128,
        per_device_batch_size=8,
        eval_steps=4,
        save_steps=32,
        save_total_limit=32,
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
def main() -> None:
    """Run the fixed 1,024-step Qwen3-0.6B experiment; see the adjacent README."""
    build_trainer(experiment_config()).train()


if __name__ == "__main__":
    main()
