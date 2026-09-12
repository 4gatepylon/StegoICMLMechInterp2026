import math
from types import SimpleNamespace

import pytest
import torch

from ciphers.kirchenbauer_et_al.binary_classification_mvp.extract import probability_of_bit


GREEN = torch.tensor([0, 1])
RED = torch.tensor([2, 3])
DELTA = math.log(3)
TOKEN_IDS = {
    "all green": [2, 0, 1, 0],
    "all red": [0, 2, 3, 2],
    "two green one red": [2, 0, 1, 2],
    "two red one green": [0, 2, 3, 0],
    "balanced": [0, 2, 0],
    "one red": [0, 2],
}


class Tokenizer:
    def __call__(self, text: str, return_tensors: str) -> SimpleNamespace:
        assert return_tensors == "pt"
        return SimpleNamespace(input_ids=torch.tensor([TOKEN_IDS[text]]))


class FixedBaseModel(torch.nn.Module):
    def __init__(self, probabilities: list[float]) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(len(probabilities), 1)
        self.register_buffer("token_logits", torch.tensor(probabilities).log())

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embedding

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        logits = self.token_logits.expand(input_ids.shape[0], input_ids.shape[1], -1)
        return SimpleNamespace(logits=logits)


@pytest.fixture
def uniform_model() -> FixedBaseModel:
    return FixedBaseModel([0.25, 0.25, 0.25, 0.25])


def probability(text: str, bit: int, model: FixedBaseModel) -> float:
    return probability_of_bit(text, bit, model, Tokenizer(), RED, GREEN, DELTA)


def test_all_red_and_all_green_have_equal_opposite_posteriors(uniform_model: FixedBaseModel) -> None:
    expected_correct = 27 / 28  # Three same-color tokens with an exp(delta) = 3 likelihood ratio each.

    assert probability("all red", 1, uniform_model) == pytest.approx(expected_correct)
    assert probability("all red", 0, uniform_model) == pytest.approx(1 - expected_correct)
    assert probability("all green", 0, uniform_model) == pytest.approx(expected_correct)
    assert probability("all green", 1, uniform_model) == pytest.approx(1 - expected_correct)


@pytest.mark.parametrize(
    ("text", "bit", "expected"),
    [
        ("two red one green", 1, 3 / 4),
        ("two green one red", 0, 3 / 4),
        ("balanced", 0, 1 / 2),
        ("balanced", 1, 1 / 2),
    ],
)
def test_small_uniform_cases_have_known_posteriors(text: str, bit: int, expected: float, uniform_model: FixedBaseModel) -> None:
    assert probability(text, bit, uniform_model) == pytest.approx(expected)


def test_base_color_mass_normalizes_the_posterior() -> None:
    model = FixedBaseModel([0.45, 0.45, 0.05, 0.05])

    # For one observed RED token, Z_green = 2.8 and Z_red = 1.2. Thus the
    # likelihoods after cancelling the common base probability are 1/2.8 and
    # 3/1.2, whose normalized RED posterior is 0.875 rather than count-only 0.75.
    assert probability("one red", 1, model) == pytest.approx(0.875)
