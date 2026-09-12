"""Continue-pretrain Qwen3-4B-Base on FineWeb with no-encoding prefixes.

It holds out validation documents and lazily prepends fresh random fixed-width bits.
LoRA minimizes next-token NLL on prefix and text so loss is low with
`do_encoding=no`; no KL or red/green boosting is used.
"""

import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ciphers.kirchenbauer_et_al.binary_classification_mvp.data import load_fineweb, prefix_batch  # noqa: E402

N_BITS = 8
N_VALIDATION_SAMPLES = 1_000


def main() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if 32 % world_size:
        raise ValueError(f"WORLD_SIZE={world_size} cannot produce an exact global batch size of 32")
    per_device_batch_size = min(8, 32 // world_size)
    dataset = load_fineweb()

    add_prefix = lambda batch: {"text": prefix_batch(batch["text"], N_BITS, False)[0]}
    train_dataset = dataset.skip(N_VALIDATION_SAMPLES).map(add_prefix, batched=True)
    validation_dataset = dataset.take(N_VALIDATION_SAMPLES).map(add_prefix, batched=True)
    trainer = SFTTrainer(
        model="Qwen/Qwen3-4B-Base",
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=LoraConfig(task_type="CAUSAL_LM", r=32, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear"),
        args=SFTConfig(
            output_dir=os.path.join(os.environ["STEGO_ARTIFACTS_DIR"], "qwen3-4b-fineweb-no-encoding-prefix-pretrain-lora"),
            run_name="qwen3-4b-fineweb-no-encoding-prefix-pretrain-lora",
            report_to="wandb",
            max_steps=10_000,
            # Qwen3 S1 uses length 4,096 (Technical Report, p. 4): https://arxiv.org/pdf/2505.09388
            max_length=4_096,
            loss_type="nll",
            # Batch 32 and LR 3e-4: NVIDIA Qwen3-4B recipe: https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/apidocs/bridge/bridge.recipes.qwen.qwen3_4b.html
            # NOTE: Qwen does not disclose Qwen3-4B's numeric batch size; 32 is NVIDIA's recipe default, not Qwen's reported pretraining batch.
            # These proxy hyperparameters were selected by Codex (AI), not by a human.
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=32 // (per_device_batch_size * world_size),
            learning_rate=3e-4,
            warmup_steps=300,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=500,
            save_steps=500,
            save_total_limit=2,
            model_init_kwargs={"torch_dtype": torch.bfloat16},
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
