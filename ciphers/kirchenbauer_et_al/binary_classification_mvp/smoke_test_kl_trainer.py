"""Run one CPU step through PrefixKLTrainer with a tiny random Qwen model."""

import math
import os
import random

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from trl import SFTConfig

from ciphers.kirchenbauer_et_al.binary_classification_mvp.kl_trainer import PrefixKLTrainer, text_collator


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")
    tokenizer.pad_token = tokenizer.eos_token
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
                         pad_token_id=tokenizer.pad_token_id)
    for loss_mode in ("nll", "ignore_prefix"):
        random.seed(0)  # The two examples deterministically exercise both gate values.
        trainer = PrefixKLTrainer(
            model=Qwen3ForCausalLM(config), loss_mode=loss_mode, processing_class=tokenizer,
            train_dataset=Dataset.from_dict({"text": ["A short training example.", "Another pretraining document."]}),
            data_collator=text_collator,
            peft_config=LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=["q_proj", "v_proj"]),
            args=SFTConfig(
                output_dir=os.path.join(os.environ["STEGO_ARTIFACTS_DIR"], f"prefix-kl-smoke-test-{loss_mode}"),
                max_length=48, max_steps=1, per_device_train_batch_size=2, gradient_checkpointing=False,
                report_to="none", save_strategy="no", remove_unused_columns=False,
                dataset_kwargs={"skip_prepare_dataset": True},
            ),
        )
        assert math.isfinite(trainer.train().training_loss)


if __name__ == "__main__":
    main()
