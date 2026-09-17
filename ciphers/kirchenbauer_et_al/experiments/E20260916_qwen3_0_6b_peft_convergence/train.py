"""Ablate Qwen3 base-model size and the prefix-KL objective with compact LoRA saves."""

import os
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
from ciphers.kirchenbauer_et_al.src.trainer_kl_fineweb import PrefixKLTrainer

QwenModel = Literal["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base"]
LossType = Literal["nll", "ignore_prefix"]
WANDB_PROJECT = "E20260916_qwen3_peft_convergence"
DATA_LENGTH = 1024
# Default data-token budget; prefixes add model-input overhead.
NUM_TRAINING_TOKENS = 128 * 256 * DATA_LENGTH


class FixedBudgetTrainingConfig(PrefixKLTrainingConfig):
    """Derive training steps from padded data positions, excluding prefixes."""

    model: QwenModel = "Qwen/Qwen3-0.6B-Base"
    n_bits: int = Field(default=1, ge=1, le=4)
    min_gpt2_document_tokens: int = Field(default=756, ge=0)
    min_qwen_document_tokens: int = Field(default=1024, ge=0)
    data_length: int = Field(default=DATA_LENGTH, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0, allow_inf_nan=False)
    alpha: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    delta: float = Field(default=2.0, ge=0, allow_inf_nan=False)
    num_training_tokens: int = Field(default=NUM_TRAINING_TOKENS, gt=0)

    @model_validator(mode="after")
    def derive_run_identity(self) -> Self:
        """Return settings with a shared project and a descriptive run/output name.

        Model size, bits, learning rate, global batch, loss, alpha, delta, token
        budget, and active document-length bounds distinguish ablations. Repeats
        with identical settings reuse the name;
        use a fresh STEGO_ARTIFACTS_DIR for independent checkpoint outputs.
        The explicit project overrides WANDB_PROJECT through build_trainer.
        """
        model_name = self.model.split("/")[1].removesuffix("-Base").lower()
        self.wandb_project = WANDB_PROJECT
        # Preserve float precision so nearby ablation settings cannot share a path.
        lr, alpha, delta = (str(value).removesuffix(".0") for value in (self.learning_rate, self.alpha, self.delta))
        self.run_name = f"{model_name}-{self.n_bits}bit-lr{lr}-gb{self.global_batch_size}-{self.loss_mode}-a{alpha}-d{delta}-tokens{self.num_training_tokens}"
        if self.min_gpt2_document_tokens > 0 or self.max_gpt2_document_tokens is not None:
            upper = "all" if self.max_gpt2_document_tokens is None else str(self.max_gpt2_document_tokens)
            self.run_name += f"-gpt2-{self.min_gpt2_document_tokens}-{upper}"
        if self.min_qwen_document_tokens > 0 or self.max_qwen_document_tokens is not None:
            upper = "all" if self.max_qwen_document_tokens is None else str(self.max_qwen_document_tokens)
            self.run_name += f"-qwen-{self.min_qwen_document_tokens}-{upper}"
        return self

    @model_validator(mode="after")
    def derive_step_budget(self) -> Self:
        """Return settings with exact-budget steps and retention for every save.

        Reject batches whose token count cannot divide the training token budget;
        rounding would change the number of padded data positions processed.
        Retention includes the Trainer's final save for a partial save interval.
        """
        if self.global_batch_size is None or self.num_training_tokens % (self.data_length * self.global_batch_size):
            raise ValueError(
                f"num_training_tokens ({self.num_training_tokens}) must be divisible by data length ({self.data_length}) * global batch size ({self.global_batch_size})"
            )
        self.max_steps = self.num_training_tokens // self.data_length // self.global_batch_size
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
    reject_document_padding: bool = True,
    dump_inputs: int = 1,
    prepend_student_bos: bool = False,
    min_gpt2_document_tokens: int = 756,
    max_gpt2_document_tokens: int | None = None,
    min_qwen_document_tokens: int = 1024,
    max_qwen_document_tokens: int | None = None,
) -> FixedBudgetTrainingConfig:
    """Return validated ablation settings consumed by ``build_trainer``.

    ``local_batch_size`` is sequences per device per microbatch; ``global_batch_size``
    includes all devices and accumulation. ``num_training_tokens`` counts padded
    data positions, excluding prefixes and validation. Steps divide this budget by
    ``DATA_LENGTH`` and the global batch size; a non-integer result is rejected.
    ``lr`` controls optimizer update size. ``model`` selects a supported Qwen3 Base
    checkpoint; ``n_bits`` sets message length and the number of text blocks.
    ``loss_type='nll'`` minimizes prefix NLL plus ``alpha`` times data KL;
    ``'ignore_prefix'`` uses only data KL and ignores alpha. ``delta`` boosts the
    teacher's selected vocabulary logits: zero disables the boost; larger values
    strengthen the encoding target. Alpha and delta must be finite and nonnegative.
    ``min_gpt2_document_tokens`` and ``max_gpt2_document_tokens`` are inclusive cached
    GPT-2 bounds. ``min_qwen_document_tokens`` and ``max_qwen_document_tokens``
    then bound complete lengths under the selected model's tokenizer, excluding
    prefixes and special tokens. Each zero minimum/``None`` maximum disables
    that stage. Both stages precede splitting; the training budget is unchanged.
    ``reject_document_padding`` makes the trainer fail before a forward pass if
    any document is padded; disabling it permits right padding only.
    ``dump_inputs`` saves the first N training microbatches per rank; zero disables.
    ``prepend_student_bos`` prepends BOS before the student prefix; teacher BOS
    is always present. The effective run/output name appends student-bos0/1.
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
        reject_document_padding=reject_document_padding,
        dump_inputs=dump_inputs,
        prepend_student_bos=prepend_student_bos,
        dataset_cache_name="fineweb-500k",
        min_gpt2_document_tokens=min_gpt2_document_tokens,
        max_gpt2_document_tokens=max_gpt2_document_tokens,
        min_qwen_document_tokens=min_qwen_document_tokens,
        max_qwen_document_tokens=max_qwen_document_tokens,
        data_length=DATA_LENGTH,
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
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    training_args = build_sft_config(config, accumulation_steps, tokenizer)
    training_args.save_only_model = True
    configure_wandb_environment(config, os.environ)
    # W&B otherwise writes its local logs relative to the working directory.
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = training_args.output_dir
    required_documents = config.validation_samples + config.max_steps * config.per_device_batch_size * world_size * accumulation_steps
    dataset = load_fineweb_cache(
        config.dataset_cache_name,
        minimum_documents=required_documents,
        min_gpt2_document_tokens=config.min_gpt2_document_tokens,
        max_gpt2_document_tokens=config.max_gpt2_document_tokens,
        min_qwen_document_tokens=config.min_qwen_document_tokens,
        max_qwen_document_tokens=config.max_qwen_document_tokens,
        tokenizer=tokenizer,
    )
    validation_dataset = dataset.take(config.validation_samples).map(fixed_prefix_metadata, with_indices=True, fn_kwargs={"n_bits": config.n_bits})
    return PrefixKLTrainer(
        model=config.model,
        loss_mode=config.loss_mode,
        alpha=config.alpha,
        n_bits=config.n_bits,
        delta=config.delta,
        reject_document_padding=config.reject_document_padding,
        dump_inputs=config.dump_inputs,
        prepend_student_bos=config.prepend_student_bos,
        strategy=config.strategy,
        train_dataset=dataset.skip(config.validation_samples),
        eval_dataset=validation_dataset,
        data_length=config.data_length,
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
    type=click.IntRange(min=1, max=4),
    help="Message bits per sequence. More bits increase payload and divide the text into shorter blocks per bit; use 1, 2, or 4 for this sweep.",
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
    help="Total padded data positions, excluding prefixes and validation. Increase to train longer at a fixed batch size.",
)
@click.option(
    "--reject-document-padding/--allow-document-padding", default=True, show_default=True, help="Reject padded documents before model execution; left padding is always forbidden."
)
@click.option("--dump-inputs", "--dump_inputs", default=1, show_default=True, type=click.IntRange(min=0), help="Dump first N training microbatches per rank; 0 disables.")
@click.option("--prepend-student-bos/--no-prepend-student-bos", default=False, show_default=True, help="Prepend BOS before the student prefix; teacher BOS is always present.")
@click.option("--min-gpt2-document-tokens", default=756, show_default=True, type=click.IntRange(min=0), help="Inclusive minimum cached GPT-2 count; applied first.")
@click.option("--max-gpt2-document-tokens", default=None, type=click.IntRange(min=0), help="Inclusive maximum cached GPT-2 count; omitted means unlimited.")
@click.option("--min-qwen-document-tokens", default=1024, show_default=True, type=click.IntRange(min=0), help="Inclusive minimum Qwen count after GPT-2 filtering.")
@click.option("--max-qwen-document-tokens", default=None, type=click.IntRange(min=0), help="Inclusive maximum Qwen count after GPT-2 filtering; omitted means unlimited.")
def main(
    local_batch_size: int,
    global_batch_size: int,
    num_training_tokens: int,
    lr: float,
    model: QwenModel,
    n_bits: int,
    loss_type: LossType,
    alpha: float,
    delta: float,
    reject_document_padding: bool,
    dump_inputs: int,
    prepend_student_bos: bool,
    min_gpt2_document_tokens: int,
    max_gpt2_document_tokens: int | None,
    min_qwen_document_tokens: int,
    max_qwen_document_tokens: int | None,
) -> None:
    """Train for a token budget, deriving optimizer steps from sequence and batch sizes.

    Steps equal num_training_tokens / (data_length * global_batch_size).
    Data length * global batch must divide the token budget exactly.
    Global batch must be divisible by local batch * WORLD_SIZE.
    Checkpoints follow the configured save cadence; all are retained.
    """
    try:
        trainer = build_trainer(
            experiment_config(
                local_batch_size,
                global_batch_size,
                num_training_tokens,
                lr=lr,
                model=model,
                n_bits=n_bits,
                loss_type=loss_type,
                alpha=alpha,
                delta=delta,
                reject_document_padding=reject_document_padding,
                dump_inputs=dump_inputs,
                prepend_student_bos=prepend_student_bos,
                min_gpt2_document_tokens=min_gpt2_document_tokens,
                max_gpt2_document_tokens=max_gpt2_document_tokens,
                min_qwen_document_tokens=min_qwen_document_tokens,
                max_qwen_document_tokens=max_qwen_document_tokens,
            )
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    trainer.train()


if __name__ == "__main__":
    main()
