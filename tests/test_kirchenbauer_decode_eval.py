import sys
from types import SimpleNamespace

import pytest
import torch

from ciphers.kirchenbauer_et_al.binary_classification_mvp.decode_eval import DecodeEvaluationCallback, decode_bit_probabilities, decode_metrics, shard_sample_indices
from ciphers.kirchenbauer_et_al.binary_classification_mvp.train_kl_fineweb import parse_args, resolve_per_device_batch_size


def two_token_tokenizer(text: str, add_special_tokens: bool) -> dict[str, list[int]]:
    assert text and not add_special_tokens
    return {"input_ids": [0, 1]}


def test_decode_bit_probabilities_recovers_block_messages() -> None:
    observed = torch.tensor(
        [
            [0, 1, 2, 3],
            [2, 3, 0, 1],
        ]
    )
    uniform_logprobs = torch.full((2, 4, 4), -torch.log(torch.tensor(4.0)))

    probabilities = decode_bit_probabilities(observed, uniform_logprobs, n_bits=2, delta=2.0, strategy="block")

    assert probabilities[0, 0] < 0.5
    assert probabilities[0, 1] > 0.5
    assert probabilities[1, 0] > 0.5
    assert probabilities[1, 1] < 0.5


def test_decode_bit_probabilities_recovers_modulo_messages() -> None:
    observed = torch.tensor([[0, 2, 1, 3]])
    uniform_logprobs = torch.full((1, 4, 4), -torch.log(torch.tensor(4.0)))

    probabilities = decode_bit_probabilities(observed, uniform_logprobs, n_bits=2, delta=2.0, strategy="modulo")

    assert probabilities[0, 0] < 0.5
    assert probabilities[0, 1] > 0.5


def test_decode_metrics_combines_bits_and_messages() -> None:
    records = [
        (0, 0, 0.1, 0),
        (0, 1, 0.9, 1),
        (1, 0, 0.8, 0),
        (1, 1, 0.7, 1),
    ]

    metrics = decode_metrics(records)

    assert metrics["eval_decode_bit_accuracy"] == pytest.approx(0.75)
    assert metrics["eval_decode_message_accuracy"] == pytest.approx(0.5)
    assert metrics["eval_decode_auroc"] == pytest.approx(0.75)


def test_decode_metrics_counts_tied_scores_as_half_auc() -> None:
    metrics = decode_metrics([(0, 0, 0.5, 0), (1, 0, 0.5, 1)])

    assert metrics["eval_decode_auroc"] == pytest.approx(0.5)


def test_batch_size_is_derived_from_world_size_and_accumulation() -> None:
    assert resolve_per_device_batch_size(128, gradient_accumulation_steps=2, world_size=8, requested=None) == 8
    assert resolve_per_device_batch_size(128, gradient_accumulation_steps=2, world_size=8, requested=8) == 8


def test_inconsistent_batch_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="must equal global batch size"):
        resolve_per_device_batch_size(128, gradient_accumulation_steps=2, world_size=8, requested=4)


def test_decode_samples_are_sharded_without_duplicates() -> None:
    shards = [shard_sample_indices(8, process_index, num_processes=3) for process_index in range(3)]

    assert sorted(index for shard in shards for index in shard) == list(range(8))
    assert not (set(shards[0]) & set(shards[1]) or set(shards[0]) & set(shards[2]) or set(shards[1]) & set(shards[2]))


def test_training_cli_defaults_to_requested_batch_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_kl_fineweb.py"])

    args = parse_args()

    assert args.global_batch_size == 128
    assert args.gradient_accumulation_steps == 2
    assert args.learning_rate == pytest.approx(3e-4)


def test_lr_alias_is_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_kl_fineweb.py", "--lr", "1e-4"])

    assert parse_args().learning_rate == pytest.approx(1e-4)


def test_block_decode_defaults_to_full_trained_span() -> None:
    trainer = SimpleNamespace(n_bits=2, strategy="block", processing_class=two_token_tokenizer, args=SimpleNamespace(max_length=10))

    callback = DecodeEvaluationCallback(trainer, steps=4, n_samples=2, n_tokens=None, per_device_batch_size=1)

    assert callback.n_tokens == 8


def test_block_decode_rejects_partial_position_range() -> None:
    trainer = SimpleNamespace(n_bits=2, strategy="block", processing_class=two_token_tokenizer, args=SimpleNamespace(max_length=10))

    with pytest.raises(ValueError, match="must span every trained data position"):
        DecodeEvaluationCallback(trainer, steps=4, n_samples=2, n_tokens=4, per_device_batch_size=1)
