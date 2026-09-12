"""Continue pretraining Qwen3-4B-Base on a streaming FineWeb sample."""

import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ciphers.kirchenbauer_et_al.binary_classification_mvp.data import load_fineweb


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
            # Batch 32: NVIDIA Qwen3-4B recipe: https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/apidocs/bridge/bridge.recipes.qwen.qwen3_4b.html
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
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
