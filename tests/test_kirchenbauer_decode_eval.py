import pytest
import torch

from ciphers.kirchenbauer_et_al.binary_classification_mvp.decode_eval import decode_bit_probabilities, decode_metrics


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
