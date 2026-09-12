"""Continue pretraining Qwen3-4B-Base on a streaming FineWeb sample."""

import os

import torch
from data import load_fineweb
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def main() -> None:
    trainer = SFTTrainer(
        model="Qwen/Qwen3-4B-Base",
        train_dataset=load_fineweb(None),
        peft_config=LoraConfig(task_type="CAUSAL_LM", r=32, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear"),
        args=SFTConfig(
            output_dir=os.path.join(os.environ["STEGO_ARTIFACTS_DIR"], "qwen3-4b-fineweb-none-prefix-lora"),
            run_name="qwen3-4b-fineweb-none-prefix-lora",
            report_to="wandb",
            max_steps=10_000,
            max_length=2_048,
            loss_type="nll",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            warmup_steps=300,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            model_init_kwargs={"torch_dtype": torch.bfloat16},
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
