import random

import pytest
import torch

from ciphers.kirchenbauer_et_al.binary_classification_mvp.data import fixed_prefix_metadata, prefix_batch
from ciphers.kirchenbauer_et_al.binary_classification_mvp.kl_trainer import PrefixKLTrainer, prefix_bits_encoding_text_collator


def test_fixed_validation_prefix_metadata_is_reproducible() -> None:
    first_pass = [fixed_prefix_metadata({"text": "example"}, index, n_bits=8) for index in range(10)]
    second_pass = [fixed_prefix_metadata({"text": "example"}, index, n_bits=8) for index in range(10)]

    assert first_pass == second_pass
    assert all(len(example["prefix_bits"]) == 8 for example in first_pass)
    assert all(set(example["prefix_bits"]) <= {"0", "1"} for example in first_pass)


def test_collator_preserves_validation_prefix_metadata() -> None:
    examples = [
        {"text": "first", "prefix_bits": "0011", "do_encoding": False},
        {"text": "second", "prefix_bits": "1100", "do_encoding": True},
    ]

    def tokenizer(texts: list[str], max_length: int | None = None, return_tensors: str | None = None, **_) -> dict:
        ids = [list(range(max_length or 4)) for _ in texts]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.ones_like(torch.tensor(ids))} if return_tensors else {"input_ids": ids}

    batch = prefix_bits_encoding_text_collator(examples, tokenizer, n_bits=4, max_length=12)
    assert batch["prefix_bits"] == ["0011", "1100"]
    assert batch["do_encoding"] == [False, True]
    assert batch["input_ids"].shape == batch["labels"].shape == (2, 12)


def test_training_prefixes_are_resampled() -> None:
    random.seed(0)

    _, first_bits, first_gates = prefix_batch(["first", "second"], n_bits=8)
    _, second_bits, second_gates = prefix_batch(["first", "second"], n_bits=8)

    assert (first_bits, first_gates) != (second_bits, second_gates)


@pytest.mark.parametrize("collator", [None, lambda examples: examples])
def test_trainer_requires_prefix_collator(collator) -> None:
    with pytest.raises(ValueError, match="requires prefix_bits_encoding_text_collator"):
        PrefixKLTrainer(data_collator=collator)
