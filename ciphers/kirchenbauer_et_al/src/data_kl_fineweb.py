"""FineWeb loading and control-prefix preprocessing for prefix-KL training."""

import random

import torch
from datasets import IterableDataset, load_dataset
from jaxtyping import Int
from transformers import PreTrainedTokenizerBase

VALIDATION_PREFIX_SEED = 42
TokenBatch = Int[torch.Tensor, "batch tokens"]  # noqa: F722


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


def prefix_token_length(tokenizer: PreTrainedTokenizerBase, n_bits: int) -> int:
    """Measure the reference prefix width used for the model-input budget.

    Args:
        tokenizer: The same tokenizer used by the training collator, without
            added special tokens.
        n_bits: Positive binary message width. Both gate values are measured
            using an all-zero message.

    Returns:
        Prefix width in tokens, added to ``data_length`` by ``build_sft_config``.
        Gate widths must match. This probe cannot establish constant width for
        arbitrary messages: ``tokenize_with_prefix`` checks every actual prefix
        against this reference, including across separate batches.
    """
    if n_bits < 1:
        raise ValueError("n_bits must be positive")
    reference_prefixes = [compile_prefix("0" * n_bits, gate) for gate in (False, True)]
    lengths = {len(ids) for ids in tokenizer(reference_prefixes, add_special_tokens=False)["input_ids"]}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("control prefixes must have the same positive token length for both gate values")
    return lengths.pop()


def tokenize_with_prefix(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    bits: list[str],
    enabled: list[bool],
    data_length: int,
) -> tuple[dict[str, TokenBatch], dict[str, TokenBatch], int]:
    """Build prefixed and unprefixed model inputs for KL training.

    Args:
        tokenizer: Hugging Face tokenizer used by both models. Must use right
            padding; left padding would shift document positions within bit blocks.
        texts: Raw data texts, one per example.
        bits: Fixed-width binary message strings, one per text.
        enabled: Whether each example requests encoding.
        data_length: Positive number of document token slots, excluding the
            prefix. Longer documents are truncated; shorter ones are padded.

    Returns:
        ``(prefixed_model_inputs, unprefixed_model_inputs, prefix_length)``. Pass
        the first dictionary to the adapter-enabled model and the second to the
        disabled-adapter teacher. Both contain ``input_ids`` and ``attention_mask``
        with shapes ``[batch, prefix_length + data_length]`` and
        ``[batch, data_length]``. The latter IDs
        exactly equal the former IDs after ``prefix_length``, guaranteeing aligned
        teacher/student KL targets.

    NOTE: it is guaranteed that the outputs will have the right length, but THERE COULD BE PADDING
    unless you already controlled for that via i.e. data filtering.
    """
    if getattr(tokenizer, "padding_side", "right") != "right":
        raise ValueError("tokenizer.padding_side must be right; left padding is forbidden")
    if not texts or len(texts) != len(bits) or len(texts) != len(enabled):
        raise ValueError("texts, bits, and enabled must have the same nonzero length")
    if data_length < 1:
        raise ValueError("data_length must be positive")
    if len({len(bit) for bit in bits}) != 1:
        raise ValueError("messages must have the same bit width")
    prefixes: list[str] = [compile_prefix(bit, gate) for bit, gate in zip(bits, enabled)]
    prefix_ids_list: list[list[int]] = tokenizer(prefixes, add_special_tokens=False)["input_ids"]
    Q: int = prefix_token_length(tokenizer, len(bits[0]))
    if any(len(ids) != Q for ids in prefix_ids_list):
        raise ValueError("control prefix token length differs from the reference used for max_length")
    base_encoding: dict[str, torch.Tensor] = tokenizer(texts, add_special_tokens=False, max_length=data_length, truncation=True, padding="max_length", return_tensors="pt")
    unprefixed_model_inputs = {"input_ids": base_encoding["input_ids"], "attention_mask": base_encoding["attention_mask"]}
    prefix_ids: torch.Tensor = torch.tensor(prefix_ids_list)
    # NOTE: Concatenate in token space to preserve consistent whitespace tokenization:
    # joint text tokenization can merge prefix/document whitespace across the boundary,
    # changing the student's data tokens relative to the unprefixed teacher's.
    prefixed_model_inputs = {
        "input_ids": torch.cat((prefix_ids, unprefixed_model_inputs["input_ids"]), dim=1),
        "attention_mask": torch.cat((torch.ones_like(prefix_ids), unprefixed_model_inputs["attention_mask"]), dim=1),
    }
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
