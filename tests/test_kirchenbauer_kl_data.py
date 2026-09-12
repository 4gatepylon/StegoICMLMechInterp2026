import random

from ciphers.kirchenbauer_et_al.binary_classification_mvp.data import fixed_prefix_metadata, prefix_batch
from ciphers.kirchenbauer_et_al.binary_classification_mvp.kl_trainer import text_collator


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

    assert text_collator(examples) == {
        "text": ["first", "second"],
        "prefix_bits": ["0011", "1100"],
        "do_encoding": [False, True],
    }


def test_training_prefixes_are_resampled() -> None:
    random.seed(0)

    _, first_bits, first_gates = prefix_batch(["first", "second"], n_bits=8)
    _, second_bits, second_gates = prefix_batch(["first", "second"], n_bits=8)

    assert (first_bits, first_gates) != (second_bits, second_gates)
