from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from ciphers.kirchenbauer_et_al.src.trainer_kl_fineweb import (
    PrefixKLTrainer,
    divergence_ignoring_prefix,
    divergence_with_prefix_nll,
)


def loss_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student_logprobs = torch.tensor([[[1.5, -0.5], [-0.2, 0.8], [1.2, -0.3]]]).log_softmax(dim=-1)
    target_logprobs = torch.tensor([[[-0.4, 0.9], [0.6, -0.1]]]).log_softmax(dim=-1)
    prefix_targets = torch.tensor([[0]])
    return student_logprobs, target_logprobs, prefix_targets


@pytest.mark.parametrize("loss_mode, alpha", [("nll", 2.5), ("ignore_prefix", 2.5)])
def test_loss_components_reconstruct_total(loss_mode: str, alpha: float) -> None:
    trainer = SimpleNamespace(loss_mode=loss_mode, alpha=alpha)

    loss, prefix_loss, data_loss = PrefixKLTrainer._divergence(trainer, *loss_inputs(), Q=1)

    assert loss == pytest.approx((prefix_loss + data_loss).item())
    student_logprobs, target_logprobs, prefix_targets = loss_inputs()
    raw_data_kl = divergence_ignoring_prefix(student_logprobs, target_logprobs, Q=1)
    if loss_mode == "nll":
        expected = divergence_with_prefix_nll(
            student_logprobs,
            target_logprobs,
            prefix_targets,
            Q=1,
            alpha=alpha,
        )
        assert loss == pytest.approx(expected.item())
        assert data_loss == pytest.approx((alpha * raw_data_kl).item())
    else:
        assert prefix_loss.item() == 0.0
        assert data_loss == pytest.approx(raw_data_kl.item())


def test_loss_components_use_separate_train_and_eval_buffers() -> None:
    trainer = SimpleNamespace(
        model=SimpleNamespace(training=True),
        accelerator=SimpleNamespace(gather_for_metrics=lambda value: value[None]),
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
    )

    PrefixKLTrainer._record_loss_metrics(trainer, torch.tensor(1.25), torch.tensor(2.5))
    trainer.model.training = False
    PrefixKLTrainer._record_loss_metrics(trainer, torch.tensor(0.75), torch.tensor(1.5))

    assert trainer._metrics["train"] == {
        "prefix_loss": [1.25],
        "data_loss": [2.5],
    }
    assert trainer._metrics["eval"] == {
        "prefix_loss": [0.75],
        "data_loss": [1.5],
    }
