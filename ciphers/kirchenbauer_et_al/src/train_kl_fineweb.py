"""LoRA-train Qwen on FineWeb with the configurable gated prefix KL objective."""

import os
import sys
from functools import partial
from pathlib import Path

from peft import LoraConfig
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ciphers.kirchenbauer_et_al.src.cache_fineweb import load_fineweb_cache  # noqa: E402
from ciphers.kirchenbauer_et_al.src.configuration_kl_fineweb import build_sft_config, configure_wandb_environment, gradient_accumulation_steps, parse_args  # noqa: E402
from ciphers.kirchenbauer_et_al.src.data_kl_fineweb import fixed_prefix_metadata  # noqa: E402
from ciphers.kirchenbauer_et_al.src.trainer_kl_fineweb import PrefixKLTrainer, prefix_bits_encoding_text_collator  # noqa: E402


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    grad_accumulation_steps = gradient_accumulation_steps(args, world_size)
    configure_wandb_environment(args, os.environ)
    required_documents = args.validation_samples + args.max_steps * args.per_device_batch_size * world_size * grad_accumulation_steps
    dataset = load_fineweb_cache(args.dataset_cache_name, minimum_documents=required_documents)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
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
        args=build_sft_config(args, grad_accumulation_steps),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
