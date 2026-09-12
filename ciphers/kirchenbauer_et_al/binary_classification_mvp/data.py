"""FineWeb loading and control-prefix preprocessing."""

from datasets import load_dataset


def compile_prefix(bits: str | None) -> str:
    """Return the literal control prefix for arbitrary bits or no encoding."""
    if bits is not None and (not bits or set(bits) - {"0", "1"}):
        raise ValueError("bits must be a non-empty binary string or None")
    enabled, value = ("no", "none") if bits is None else ("yes", bits)
    return f"<encoding> <do_encoding> {enabled} </do_encoding> <encoding_value> {value} </encoding_value> </encoding>\n"


def load_fineweb(bits: str | None, n: int | None = None):
    """Stream shuffled FineWeb with the requested control prefix prepended."""
    prefix = compile_prefix(bits)
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
        revision="9bb295ddab0e05d785b879661af7260fed5140fc",
    ).shuffle(seed=42, buffer_size=10_000)
    # Return dataset in a format that can be used by HF SFTTrainer for training on ALL tokens.
    dataset = dataset.map(lambda row: {"text": prefix + row["text"]})
    return dataset if n is None else dataset.take(n)
