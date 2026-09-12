from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.cache import tokenizer_fingerprint
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.colors import (
    build_color_partition,
    load_or_build_color_partition,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.constants import RED_SIGNAL
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.data import (
    CorpusConfig,
    TextExample,
    load_or_build_corpus_splits,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.objectives import (
    distribution_metrics,
    distribution_metrics_batch,
    validate_probability_mass,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.training import _batches


class DummyTokenizer:
    all_special_ids = [0, 11]
    pad_token_id = 0

    def get_vocab(self) -> dict[str, int]:
        return {f"token-{token_id}": token_id for token_id in range(12)}

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        max_length = int(kwargs["max_length"])
        token_ids = [1 + (ord(character) % 10) for character in text if not character.isspace()]
        return {"input_ids": token_ids[:max_length]}


class DummyStream(list[dict[str, str]]):
    def shuffle(self, **_: object) -> DummyStream:
        return self


class MutableBackend:
    truncation: dict[str, int] | None = None

    def to_str(self) -> str:
        return json.dumps({"model": {"type": "dummy", "vocab": {"a": 1}}, "truncation": self.truncation})


class BackendTokenizer(DummyTokenizer):
    backend_tokenizer = MutableBackend()


class TokenwiseModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.logits_by_token = torch.nn.Parameter(torch.randn(vocab_size, vocab_size))

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(logits=self.logits_by_token[input_ids])


class CacheAndBatchingTests(unittest.TestCase):
    def test_batching_keeps_the_final_partial_batch(self) -> None:
        examples = tuple(TextExample((1, 2), f"{index:064x}") for index in range(5))
        batches = _batches(examples, 2)
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        self.assertEqual([example for batch in batches for example in batch], list(examples))

    def test_tokenizer_fingerprint_ignores_transient_truncation_state(self) -> None:
        tokenizer = BackendTokenizer()
        before = tokenizer_fingerprint(tokenizer)
        tokenizer.backend_tokenizer.truncation = {"max_length": 4}
        self.assertEqual(before, tokenizer_fingerprint(tokenizer))

    def test_corpus_cache_miss_then_hit_without_dataset_access(self) -> None:
        tokenizer = DummyTokenizer()
        config = CorpusConfig(
            prefix_train_tokens=3,
            encoding_train_tokens=3,
            validation_tokens=3,
            generation_prompts=2,
            sequence_length=4,
            generation_prompt_length=3,
            min_sequence_length=2,
            shuffle_buffer_size=10,
        )
        rows = DummyStream({"text": f"unique document {index} abcdef"} for index in range(12))

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.data.load_dataset",
                return_value=rows,
            ) as load_dataset:
                first = load_or_build_corpus_splits(tokenizer, config, directory)
            load_dataset.assert_called_once()

            with patch(
                "ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.data.load_dataset",
                side_effect=AssertionError("cache hits must not access FineWeb"),
            ):
                second = load_or_build_corpus_splits(tokenizer, config, directory)

            self.assertEqual(first, second)
            payload = json.loads((Path(directory) / "corpus_splits.json").read_text())
            self.assertEqual(payload["metadata"]["corpus_config"], config.__dict__)
            self.assertEqual(set(payload["splits"]), {"prefix_train", "encoding_train", "validation", "generation"})

    def test_color_partition_cache_miss_then_exact_hit(self) -> None:
        tokenizer = DummyTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            first = load_or_build_color_partition(tokenizer, 12, seed=7, cache_dir=directory)
            with patch(
                "ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.colors.build_color_partition",
                side_effect=AssertionError("cache hits must not regenerate the partition"),
            ):
                second = load_or_build_color_partition(tokenizer, 12, seed=7, cache_dir=directory)
            self.assertEqual(first, second)
            self.assertTrue((Path(directory) / "color_partition.json").is_file())

    def test_padded_batch_matches_individual_objectives_and_masks_padding(self) -> None:
        torch.manual_seed(3)
        tokenizer = DummyTokenizer()
        model = TokenwiseModel(vocab_size=12)
        partition = build_color_partition(tokenizer, 12, seed=7)
        input_batch = [(1, 2, 3, 4), (5, 6, 7)]
        prefix_ids = (8, 9)
        raw_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]])
        raw_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
        reference_logits = model(input_ids=raw_ids, attention_mask=raw_mask).logits[:, :-1].detach()

        batched = distribution_metrics_batch(
            model,
            reference_logits,
            input_batch,
            prefix_ids,
            RED_SIGNAL,
            partition,
            delta=2.0,
            device=torch.device("cpu"),
            logit_chunk_size=2,
            pad_token_id=tokenizer.pad_token_id,
        )
        individuals = [
            distribution_metrics(
                model,
                reference_logits[index : index + 1, : len(input_ids) - 1],
                input_ids,
                prefix_ids,
                RED_SIGNAL,
                partition,
                delta=2.0,
                device=torch.device("cpu"),
                logit_chunk_size=2,
            )
            for index, input_ids in enumerate(input_batch)
        ]

        expected_loss = sum(item.loss * item.token_count for item in individuals) / sum(item.token_count for item in individuals)
        self.assertTrue(torch.allclose(batched.loss, expected_loss, atol=1e-6))
        self.assertAlmostEqual(batched.expected_red, sum(item.expected_red for item in individuals), places=5)
        self.assertAlmostEqual(batched.expected_green, sum(item.expected_green for item in individuals), places=5)
        self.assertEqual(batched.token_count, 5)
        self.assertEqual(batched.sequence_count, 2)
        validate_probability_mass(batched)
        batched.loss.backward()
        self.assertIsNotNone(model.logits_by_token.grad)


if __name__ == "__main__":
    unittest.main()
