"""LoRA-train Qwen on FineWeb with the configurable gated prefix KL objective."""

import os
import sys
from functools import partial
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ciphers.kirchenbauer_et_al.binary_classification_mvp.configuration_kl_fineweb import gradient_accumulation_steps, parse_args  # noqa: E402
from ciphers.kirchenbauer_et_al.binary_classification_mvp.data_kl_fineweb import fixed_prefix_metadata, load_fineweb  # noqa: E402
from ciphers.kirchenbauer_et_al.binary_classification_mvp.trainer_kl_fineweb import PrefixKLTrainer, prefix_bits_encoding_text_collator  # noqa: E402


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
