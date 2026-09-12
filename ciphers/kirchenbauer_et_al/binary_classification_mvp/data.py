"""FineWeb loading and control-prefix preprocessing."""

import random

from datasets import load_dataset


def compile_prefix(bits: str, do_encoding: bool) -> str:
    """Return a fixed-width binary control prefix."""
    if not bits or set(bits) - {"0", "1"}:
        raise ValueError("bits must be a non-empty binary string")
    enabled = "yes" if do_encoding else "no"
    return f"<encoding> <do_encoding> {enabled} </do_encoding> <encoding_value> {bits} </encoding_value> </encoding>\n"


def prefix_batch(texts: list[str], n_bits: int, do_encoding: bool | None = None):
    """Freshly sample prefixes for one training batch."""
    assert n_bits > 0
    bits = ["".join(random.choices("01", k=n_bits)) for _ in texts]
    enabled = [random.choice((False, True)) if do_encoding is None else do_encoding for _ in texts]
    return [compile_prefix(b, on) + text for text, b, on in zip(texts, bits, enabled)], bits, enabled


def load_fineweb(n: int | None = None):
    """Stream shuffled, unmodified FineWeb."""
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
        revision="9bb295ddab0e05d785b879661af7260fed5140fc",
    ).shuffle(seed=42, buffer_size=10_000)
    return dataset if n is None else dataset.take(n)
