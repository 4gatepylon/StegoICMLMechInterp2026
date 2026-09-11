"""Experiment-wide model, dataset, signal, prefix, and path constants."""

from pathlib import Path

MODEL_NAME = "Qwen/Qwen3-4B-Base"
DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_CONFIG = "sample-10BT"
DATASET_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PREFIX_OUTPUT = EXPERIMENT_DIR / "outputs" / "prefix_adapter"
DEFAULT_ENCODING_OUTPUT = EXPERIMENT_DIR / "outputs" / "encoding_adapter"

NULL_SIGNAL = -1
RED_SIGNAL = 0
GREEN_SIGNAL = 1
SIGNAL_NAMES = {
    NULL_SIGNAL: "none",
    RED_SIGNAL: "red",
    GREEN_SIGNAL: "green",
}

PREFIXES = {
    NULL_SIGNAL: ("<encoding> <do_encoding> no </do_encoding> <encoding_value> none </encoding_value> </encoding>\n"),
    RED_SIGNAL: ("<encoding> <do_encoding> yes </do_encoding> <encoding_value> 0 </encoding_value> </encoding>\n"),
    GREEN_SIGNAL: ("<encoding> <do_encoding> yes </do_encoding> <encoding_value> 1 </encoding_value> </encoding>\n"),
}
