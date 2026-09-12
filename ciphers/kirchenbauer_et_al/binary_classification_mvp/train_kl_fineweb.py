"""LoRA-train Qwen on FineWeb with the configurable gated prefix KL objective."""

import argparse
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig
from trl import SFTConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ciphers.kirchenbauer_et_al.binary_classification_mvp.data import fixed_prefix_metadata, load_fineweb  # noqa: E402
from ciphers.kirchenbauer_et_al.binary_classification_mvp.kl_trainer import PrefixKLTrainer, text_collator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--model", default="Qwen/Qwen3-4B-Base")
    add("--run-name", default="qwen3-4b-fineweb-prefix-kl-lora")
    add("--loss-mode", choices=("nll", "ignore_prefix"), default="nll")
    add("--strategy", choices=("block", "modulo"), default="block")
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
    add("--logging-steps", type=int, default=1)
    add("--lora-rank", type=int, default=32)
    add("--lora-alpha", type=int, default=16)
    add("--lora-dropout", type=float, default=0.05)
    add("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    add("--report-to", default="wandb")
    add("--wandb-project")
    add("--resume-from-checkpoint")
    return parser.parse_args()


def gradient_accumulation_steps(args: argparse.Namespace, world_size: int) -> int:
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
        train_dataset=dataset.skip(args.validation_samples),
        eval_dataset=validation_dataset,
        data_collator=text_collator,
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
            save_total_limit=2,
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            model_init_kwargs={"torch_dtype": getattr(torch, args.dtype)},
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
