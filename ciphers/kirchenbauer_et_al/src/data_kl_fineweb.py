"""FineWeb loading and control-prefix preprocessing for prefix-KL training."""

import random
from typing import Literal

import torch
from datasets import IterableDataset, load_dataset
from transformers import PreTrainedTokenizerBase

VALIDATION_PREFIX_SEED = 42


def compile_prefix(bits: str, do_encoding: bool) -> str:
    """Return a fixed-width binary control prefix."""
    if not bits or set(bits) - {"0", "1"}:
        raise ValueError("bits must be a non-empty binary string")
    enabled = "yes" if do_encoding else "no"
    return f"<encoding> <do_encoding> {enabled} </do_encoding> <encoding_value> {bits} </encoding_value> </encoding>\n"


def prefix_batch(texts: list[str], n_bits: int, do_encoding: bool | None = None) -> tuple[list[str], list[str], list[bool]]:
    """Freshly sample prefixes for one training batch."""
    assert n_bits > 0
    bits = ["".join(random.choices("01", k=n_bits)) for _ in texts]
    enabled = [random.choice((False, True)) if do_encoding is None else do_encoding for _ in texts]
    return [compile_prefix(b, on) + text for text, b, on in zip(texts, bits, enabled)], bits, enabled


def tokenize_with_prefix(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    bits: list[str],
    enabled: list[bool],
    max_length: int,
    concatenation_space: Literal["token", "character"] = "token",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
    """Build prefixed and unprefixed model inputs for KL training.

    Args:
        tokenizer: Hugging Face tokenizer used by both models.
        texts: Raw data texts, one per example.
        bits: Fixed-width binary message strings, one per text.
        enabled: Whether each example requests encoding.
        max_length: Padded length of each prefixed sequence.
        concatenation_space: Concatenate prefix/data token IDs in ``"token"``
            mode, or tokenize the concatenated strings in ``"character"`` mode.

    Returns:
        ``(prefixed_model_inputs, unprefixed_model_inputs, prefix_length)``. Pass
        the first dictionary to the adapter-enabled model and the second to the
        disabled-adapter teacher. Both contain ``input_ids`` and ``attention_mask``
        with shapes ``[batch, max_length]`` and
        ``[batch, max_length - prefix_length]``. In token mode, the latter IDs
        exactly equal the former IDs after ``prefix_length``, guaranteeing aligned
        teacher/student KL targets.
    """
    if not texts or len(texts) != len(bits) or len(texts) != len(enabled):
        raise ValueError("texts, bits, and enabled must have the same nonzero length")
    prefixes = [compile_prefix(bit, gate) for bit, gate in zip(bits, enabled)]
    prefix_ids = tokenizer(prefixes, add_special_tokens=False)["input_ids"]
    assert len({len(ids) for ids in prefix_ids}) == 1
    Q, M = len(prefix_ids[0]), max_length - len(prefix_ids[0])
    assert M > 0
    base_encoding = tokenizer(texts, add_special_tokens=False, max_length=M, truncation=True, padding="max_length", return_tensors="pt")
    unprefixed_model_inputs = {"input_ids": base_encoding["input_ids"], "attention_mask": base_encoding["attention_mask"]}
    prefix_ids = torch.tensor(prefix_ids)
    if concatenation_space == "token":
        prefixed_model_inputs = {
            "input_ids": torch.cat((prefix_ids, unprefixed_model_inputs["input_ids"]), dim=1),
            "attention_mask": torch.cat((torch.ones_like(prefix_ids), unprefixed_model_inputs["attention_mask"]), dim=1),
        }
        assert torch.equal(prefixed_model_inputs["input_ids"][:, Q:], unprefixed_model_inputs["input_ids"])
    elif concatenation_space == "character":
        encoding = tokenizer(
            [prefix + text for prefix, text in zip(prefixes, texts)], add_special_tokens=False, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        prefixed_model_inputs = {"input_ids": encoding["input_ids"], "attention_mask": encoding["attention_mask"]}
    else:
        raise ValueError("concatenation_space must be 'token' or 'character'")
    return prefixed_model_inputs, unprefixed_model_inputs, Q


def fixed_prefix_metadata(example: dict[str, str], index: int, n_bits: int, seed: int = VALIDATION_PREFIX_SEED) -> dict[str, str | bool]:
    """Attach reproducible prefix controls to one validation example."""
    del example
    rng = random.Random((seed << 32) + index)
    return {
        "prefix_bits": "".join(rng.choices("01", k=n_bits)),
        "do_encoding": rng.choice((False, True)),
    }


def load_fineweb(n: int | None = None) -> IterableDataset:
    """Stream shuffled, unmodified FineWeb."""
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
        revision="9bb295ddab0e05d785b879661af7260fed5140fc",
    ).shuffle(seed=42, buffer_size=10_000)
    return dataset if n is None else dataset.take(n)
