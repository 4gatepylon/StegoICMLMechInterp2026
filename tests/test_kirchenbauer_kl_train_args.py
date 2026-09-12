# TODO(hadriano): Replace argparse.Namespace after the corresponding CLI migrates to Click.
import argparse
import sys

import pytest

from ciphers.kirchenbauer_et_al.binary_classification_mvp.train_kl_fineweb import (
    gradient_accumulation_steps,
    parse_args,
)


def test_optimization_knob_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_kl_fineweb.py", "--lr", "0.001", "--batch-size", "4", "--grad-accum-steps", "8"],
    )

    args = parse_args()

    assert args.learning_rate == 0.001
    assert args.per_device_batch_size == 4
    assert gradient_accumulation_steps(args, world_size=2) == 8


def test_gradient_accumulation_defaults_to_global_batch_size() -> None:
    args = argparse.Namespace(
        per_device_batch_size=2,
        gradient_accumulation_steps=None,
        global_batch_size=None,
    )

    assert gradient_accumulation_steps(args, world_size=4) == 4


def test_gradient_accumulation_rejects_conflicting_knobs() -> None:
    args = argparse.Namespace(
        per_device_batch_size=2,
        gradient_accumulation_steps=4,
        global_batch_size=32,
    )

    with pytest.raises(ValueError, match="either global batch size or gradient accumulation steps"):
        gradient_accumulation_steps(args, world_size=1)
