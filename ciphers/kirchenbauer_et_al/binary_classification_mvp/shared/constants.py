"""Experiment-wide model, dataset, signal, prefix, and path constants."""

from pathlib import Path

MODEL_NAME = "Qwen/Qwen3-4B-Base"
DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_CONFIG = "sample-10BT"
DATASET_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
ARTIFACTS_DIR_ENV = "ARTIFACTS_DIR"

NULL_SIGNAL = -1
RED_SIGNAL = 0
GREEN_SIGNAL = 1
SIGNAL_NAMES = {
    NULL_SIGNAL: "none",
    RED_SIGNAL: "red",
    GREEN_SIGNAL: "green",
}


def compile_prefix(bits: str | None) -> str:
    """Compile a non-empty binary string, or the original-policy null signal."""

    if bits is not None:
        if not isinstance(bits, str):
            raise TypeError(f"bits must be a binary string or None, got {type(bits).__name__}")
        if not bits or any(bit not in "01" for bit in bits):
            raise ValueError(f"bits must be a non-empty binary string, got {bits!r}")
    do_encoding = "no" if bits is None else "yes"
    encoding_value = "none" if bits is None else bits
    return f"<encoding> <do_encoding> {do_encoding} </do_encoding> <encoding_value> {encoding_value} </encoding_value> </encoding>\n"


# This experiment is one-bit by default; compile_prefix also supports future
# multi-bit experiments without changing the text protocol.
PREFIXES = {
    NULL_SIGNAL: compile_prefix(None),
    RED_SIGNAL: compile_prefix("0"),
    GREEN_SIGNAL: compile_prefix("1"),
}
