"""Run one CPU step through PrefixKLTrainer with a tiny random Qwen model."""

import math
import os
import random
from itertools import product

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoConfig, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from trl import SFTConfig

from ciphers.kirchenbauer_et_al.src.data_kl_fineweb import prefix_token_length
from ciphers.kirchenbauer_et_al.src.trainer_kl_fineweb import PrefixKLTrainer


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")
    base_config = AutoConfig.from_pretrained("Qwen/Qwen3-4B-Base")
    tokenizer.pad_token = tokenizer.eos_token
    data_length = 16
    max_length = data_length + prefix_token_length(tokenizer, n_bits=8)
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=base_config.bos_token_id,
    )
    dataset = Dataset.from_dict({"text": ["A short training example. " * 16, "Another pretraining document. " * 16]})
    for loss_mode, prepend_student_bos in product(("nll", "ignore_prefix"), (False, True)):
        random.seed(0)  # The two examples deterministically exercise both gate values.
        trainer = PrefixKLTrainer(
            model=Qwen3ForCausalLM(Qwen3Config.from_dict(config.to_dict())),
            loss_mode=loss_mode,
            prepend_student_bos=prepend_student_bos,
            processing_class=tokenizer,
            train_dataset=dataset,
            eval_dataset=dataset,
            data_length=data_length,
            peft_config=LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=["q_proj", "v_proj"]),
            args=SFTConfig(
                output_dir=os.path.join(os.environ["STEGO_ARTIFACTS_DIR"], f"prefix-kl-smoke-test-{loss_mode}-student-bos{int(prepend_student_bos)}"),
                max_length=max_length + int(prepend_student_bos),
                max_steps=1,
                use_cpu=True,
                bf16=False,
                per_device_train_batch_size=2,
                gradient_checkpointing=False,
                report_to="none",
                save_strategy="no",
                prediction_loss_only=True,
                remove_unused_columns=False,
                dataset_kwargs={"skip_prepare_dataset": True},
                loss_type="nll",
            ),
        )
        batch = trainer.data_collator([dataset[0]])
        assert batch["input_ids"].shape[1] == trainer.args.max_length
        assert trainer.bos_token_id == base_config.bos_token_id
        if prepend_student_bos:
            assert batch["input_ids"][0, 0] == base_config.bos_token_id
        assert not trainer.model_accepts_loss_kwargs
        assert math.isfinite(trainer.train().training_loss)
        assert math.isfinite(trainer.evaluate()["eval_loss"])


if __name__ == "__main__":
    main()
