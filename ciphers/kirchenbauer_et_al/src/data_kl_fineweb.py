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


def initial_context_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """Return the shared initial context for training and extraction.

    ``tokenizer`` must expose BOS or EOS. Prefer BOS, falling back to EOS for
    Qwen Base tokenizers without BOS. This token lets the teacher predict the
    first document token and the student predict the first prefix token; it is
    context only, never a loss target or part of the message's data partitions.
    """
    token_id = tokenizer.bos_token_id
    if token_id is None:
        token_id = tokenizer.eos_token_id
    if token_id is None:
        raise ValueError("prefix KL requires a BOS or EOS token for initial context")
    return token_id


def tokenize_with_prefix(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    bits: list[str],
    enabled: list[bool],
    data_length: int,
) -> tuple[dict[str, TokenBatch], dict[str, TokenBatch], int]:
    """Build prefixed and unprefixed model inputs for KL training.

    Args:
        tokenizer: Hugging Face tokenizer used by both models.
        texts: Raw data texts, one per example.
        bits: Fixed-width binary message strings, one per text.
        enabled: Whether each example requests encoding.
        data_length: Positive number of document token slots, excluding the
            prefix and initial context token. Longer documents are truncated; shorter ones are padded.

    Returns:
        ``(prefixed_model_inputs, unprefixed_model_inputs, prefix_length)``. Pass
        the first dictionary to the adapter-enabled model and the second to the
        disabled-adapter teacher. Both contain ``input_ids`` and ``attention_mask``
        with shapes ``[batch, 1 + prefix_length + data_length]`` and
        ``[batch, 1 + data_length]``. Both start with the same initial context
        token (BOS, or EOS when BOS is absent); it is not included in the returned
        ``prefix_length``. Data IDs after the context/prefix are identical.
        Drop each forward pass's final logit to predict exactly the prefix/data
        tokens, rather than a token beyond the input.
    """
    if not texts or len(texts) != len(bits) or len(texts) != len(enabled):
        raise ValueError("texts, bits, and enabled must have the same nonzero length")
    if data_length < 1:
        raise ValueError("data_length must be positive")
    if len({len(bit) for bit in bits}) != 1:
        raise ValueError("messages must have the same bit width")
    prefixes = [compile_prefix(bit, gate) for bit, gate in zip(bits, enabled)]
    prefix_ids = tokenizer(prefixes, add_special_tokens=False)["input_ids"]
    Q = prefix_token_length(tokenizer, len(bits[0]))
    if any(len(ids) != Q for ids in prefix_ids):
        raise ValueError("control prefix token length differs from the reference used for max_length")
    base_encoding = tokenizer(texts, add_special_tokens=False, max_length=data_length, truncation=True, padding="max_length", return_tensors="pt")
    context_ids = torch.full_like(base_encoding["input_ids"][:, :1], initial_context_token_id(tokenizer))
    unprefixed_model_inputs = {
        "input_ids": torch.cat((context_ids, base_encoding["input_ids"]), dim=1),
        "attention_mask": torch.cat((torch.ones_like(context_ids), base_encoding["attention_mask"]), dim=1),
    }
    prefix_ids = torch.cat((context_ids, torch.tensor(prefix_ids)), dim=1)
    prefixed_model_inputs = {
        "input_ids": torch.cat((prefix_ids, base_encoding["input_ids"]), dim=1),
        "attention_mask": torch.cat((torch.ones_like(prefix_ids), base_encoding["attention_mask"]), dim=1),
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
