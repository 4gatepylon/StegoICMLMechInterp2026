import random
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from ciphers.kirchenbauer_et_al.src.data_kl_fineweb import compile_prefix, fixed_prefix_metadata, prefix_batch, prefix_token_length, tokenize_with_prefix
from ciphers.kirchenbauer_et_al.src.trainer_kl_fineweb import PrefixKLTrainer, prefix_bits_encoding_text_collator


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

    batch = prefix_bits_encoding_text_collator(examples, tokenizer, n_bits=4, data_length=8)
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


@pytest.fixture
def length_tokenizer() -> PreTrainedTokenizerFast:
    """Build an offline tokenizer with an odd prefix width to expose subtraction bugs.

    Whitespace separates data words. Splitting the opening gate tag makes the prefix
    nine tokens, while every binary message is one word. This deliberately
    differs from Qwen; the contract must work independently of prefix parity.
    """
    backend = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Split("<do_encoding", behavior="isolated")])
    backend.train_from_iterator(
        [compile_prefix("0", False), compile_prefix("1", True), "a b c"],
        trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"]),
    )
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")


@pytest.mark.parametrize("n_bits", [1, 2, 4, 8])
@pytest.mark.parametrize("document_length", [3, 4096, 4100])
@pytest.mark.parametrize("strategy", ["block", "modulo"])
def test_data_budget_and_partitions_exclude_prefix(length_tokenizer, n_bits, document_length, strategy) -> None:
    """Cover all experiment bit widths, short/exact/long data, both partitions.

    Both gates share a batch. Check truncation, padding masks, teacher/student
    alignment, and complete equal partitions despite odd prefix width. Model
    execution and real Qwen tokenization are omitted.
    """
    text = " ".join("abc"[position % 3] for position in range(document_length))
    examples = [
        {"text": text, "prefix_bits": "0" * n_bits, "do_encoding": False},
        {"text": text, "prefix_bits": "1" * n_bits, "do_encoding": True},
    ]
    data_length = 4096
    batch = prefix_bits_encoding_text_collator(examples, length_tokenizer, n_bits, data_length)
    prefix_length = batch["prefix_length"]
    assert prefix_length % 2 == 1  # Ensure this fixture would catch subtracting Q before partitioning.
    assert batch["input_ids"].shape == (2, data_length + prefix_length)
    assert batch["base_input_ids"].shape == (2, data_length)
    assert torch.equal(batch["input_ids"][:, prefix_length:], batch["base_input_ids"])
    assert torch.equal(batch["attention_mask"][:, prefix_length:], batch["base_attention_mask"])
    expected_ids = length_tokenizer(text, add_special_tokens=False)["input_ids"][:data_length]
    assert batch["base_input_ids"][0, : len(expected_ids)].tolist() == expected_ids
    assert batch["base_attention_mask"].sum(dim=1).tolist() == [min(document_length, data_length)] * 2
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
    trainer = SimpleNamespace(n_bits=n_bits, strategy=strategy)
    parts = [PrefixKLTrainer._positions(trainer, part, batch["base_input_ids"].shape[1], torch.device("cpu")) for part in range(n_bits)]
    assert all(len(part) == data_length // n_bits for part in parts)
    assert torch.equal(torch.cat(parts).sort().values, torch.arange(data_length))


@pytest.mark.parametrize("data_length,n_bits", [(0, 8), (-8, 8), (4095, 8), (4096, 0)])
def test_collator_rejects_invalid_data_partition(length_tokenizer, data_length, n_bits) -> None:
    """Cover nonpositive budgets/bit counts and nondivisible data; model execution is omitted."""
    with pytest.raises(ValueError, match="data_length"):
        prefix_bits_encoding_text_collator([{"text": "a"}], length_tokenizer, n_bits, data_length)


@pytest.mark.parametrize("mismatch", ["gate", "message"])
def test_prefix_width_changes_fail_before_model_execution(length_tokenizer, mismatch) -> None:
    """Cover gate-dependent and cross-batch message-dependent prefix widths, without a model."""

    def variable_width_tokenizer(texts, **kwargs):
        encoded = length_tokenizer(texts, **kwargs)
        for text, ids in zip(texts, encoded["input_ids"]):
            if " yes " in text if mismatch == "gate" else " 11111111 " in text:
                ids.append(length_tokenizer.unk_token_id)
        return encoded

    with pytest.raises(ValueError, match="prefix"):
        tokenize_with_prefix(variable_width_tokenizer, ["a"], ["11111111"], [True], data_length=4096)


def test_reference_length_matches_both_gate_prefixes(length_tokenizer) -> None:
    """Cover reference-to-batch agreement; exhaustive Qwen messages are checked separately."""
    reference_length = prefix_token_length(length_tokenizer, n_bits=8)
    batch = prefix_bits_encoding_text_collator([{"text": "a"}], length_tokenizer, n_bits=8, data_length=16)
    assert batch["input_ids"].shape[1] == 16 + reference_length


def test_token_concatenation_preserves_boundary_whitespace() -> None:
    """Cover a BPE newline merge across the prefix/data boundary, without Qwen or model IO."""
    vocab = {token: index for index, token in enumerate(["[UNK]", "[PAD]", *sorted(pre_tokenizers.ByteLevel.alphabet()), "ĊĊ", "ye", "yes", "no"])}
    backend = Tokenizer(models.BPE(vocab, merges=[("Ċ", "Ċ"), ("y", "e"), ("ye", "s"), ("n", "o")], unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    prefixes = [compile_prefix("0", False)]
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    text = "\nhello"
    separate_ids = tokenizer(prefixes[0], add_special_tokens=False)["input_ids"] + tokenizer(text, add_special_tokens=False)["input_ids"]
    assert tokenizer(prefixes[0] + text, add_special_tokens=False)["input_ids"] != separate_ids
    prefixed, base, prefix_length = tokenize_with_prefix(tokenizer, [text], ["0"], [False], data_length=8)
    assert prefixed["input_ids"][0, :len(separate_ids)].tolist() == separate_ids
    assert torch.equal(prefixed["input_ids"][:, prefix_length:], base["input_ids"])
