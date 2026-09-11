"""Network-free CPU integration coverage for the distillation pipeline."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

import shared  # noqa: E402


class FakeStream:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def shuffle(self, *, seed: int, buffer_size: int) -> FakeStream:
        if seed != 7 or buffer_size != 10:
            raise AssertionError("unexpected fake shuffle configuration")
        return self

    def __iter__(self):
        return iter(self.rows)


class TinyTokenizer:
    all_special_ids = [7]
    pad_token_id = 7
    eos_token_id = 7

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_attention_mask: bool = True,
    ) -> dict[str, list[int]]:
        del return_attention_mask
        if add_special_tokens:
            raise AssertionError("the experiment must not add special tokens")
        input_ids = [ord(character) % 7 for character in text]
        if truncation:
            input_ids = input_ids[:max_length]
        return {"input_ids": input_ids}

    def save_pretrained(self, output_dir: Path) -> None:
        (Path(output_dir) / "tiny_tokenizer.saved").touch()


class TinyCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 6)
        self.output = torch.nn.Linear(6, 8, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))

    def save_pretrained(self, output_dir: Path) -> None:
        (Path(output_dir) / "tiny_model.saved").touch()


class CpuIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = TinyTokenizer()
        self.corpus_config = shared.CorpusConfig(
            prefix_train_tokens=5,
            encoding_train_tokens=5,
            validation_tokens=5,
            generation_prompts=2,
            sequence_length=6,
            generation_prompt_length=4,
            min_sequence_length=3,
            dataset_seed=7,
            shuffle_buffer_size=10,
        )

    def build_splits(self) -> shared.CorpusSplits:
        rows = [{"text": chr(65 + index) * 20} for index in range(20)]
        with patch("shared.data.load_dataset", return_value=FakeStream(rows)):
            return shared.build_corpus_splits(self.tokenizer, self.corpus_config)

    def test_disjoint_corpus_and_metrics(self) -> None:
        splits = self.build_splits()
        splits.assert_disjoint()
        hashes = [
            example.source_hash
            for split in (
                splits.prefix_train,
                splits.encoding_train,
                splits.validation,
                splits.generation,
            )
            for example in split
        ]
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertEqual(
            shared.binary_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]),
            1.0,
        )
        self.assertEqual(shared.binary_auroc([0, 1], [0.5, 0.5]), 0.5)

    def test_cpu_distillation_step(self) -> None:
        splits = self.build_splits()
        reference = TinyCausalLM()
        student = copy.deepcopy(reference)
        reference.requires_grad_(False)
        partition = shared.build_color_partition(self.tokenizer, 8, seed=42)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            shared.train_distillation(
                reference_model=reference,
                student_model=student,
                train_examples=splits.encoding_train,
                validation_examples=splits.validation,
                signals=(
                    shared.RED_SIGNAL,
                    shared.GREEN_SIGNAL,
                    shared.NULL_SIGNAL,
                ),
                tokenizer=self.tokenizer,
                partition=partition,
                output_dir=output_dir,
                device=torch.device("cpu"),
                delta=2.0,
                learning_rate=1e-3,
                weight_decay=0.0,
                max_grad_norm=1.0,
                epochs=1,
                training_seed=1,
                logit_chunk_size=2,
                log_every_steps=1,
                eval_every_steps=0,
                max_eval_sequences=1,
                max_steps=1,
            )
            records = [json.loads(line) for line in (output_dir / "training_metrics.jsonl").read_text().splitlines()]
            train_records = [record for record in records if record["split"] == "train"]
            self.assertEqual(
                {record["signal"] for record in train_records},
                {"red", "green", "none"},
            )
            null_record = next(record for record in train_records if record["signal"] == "none")
            self.assertAlmostEqual(null_record["kl"], 0.0, places=6)
            self.assertTrue((output_dir / "tiny_model.saved").is_file())
            self.assertTrue((output_dir / "tiny_tokenizer.saved").is_file())

    def test_tiny_qwen_lora_checkpoint_handoff(self) -> None:
        from transformers import Qwen3Config, Qwen3ForCausalLM

        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base_path = root / "base"
            adapter_path = root / "adapter"
            Qwen3ForCausalLM(config).save_pretrained(base_path)
            stage_one, _ = shared.load_trainable_lora_model(
                str(base_path),
                torch.float32,
                lora_rank=2,
                lora_alpha=4,
                lora_dropout=0.0,
            )
            stage_one.save_pretrained(adapter_path)
            stage_two, base_name = shared.load_trainable_lora_model(
                str(adapter_path),
                torch.float32,
                lora_rank=99,
                lora_alpha=99,
                lora_dropout=0.0,
            )
            self.assertEqual(Path(base_name), base_path)
            trainable_names = [name for name, parameter in stage_two.named_parameters() if parameter.requires_grad]
            self.assertTrue(trainable_names)
            self.assertTrue(all("lora_" in name for name in trainable_names))


if __name__ == "__main__":
    unittest.main()
