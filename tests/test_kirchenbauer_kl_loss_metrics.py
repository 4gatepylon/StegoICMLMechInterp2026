from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from ciphers.kirchenbauer_et_al.src.trainer_kl_fineweb import (
    PrefixKLTrainer,
    free_token_kl,
    prefix_nll,
)

StudentLogprobs = Float[torch.Tensor, "batch prefixed_tokens vocab"]  # noqa: F722
TargetLogprobs = Float[torch.Tensor, "batch free_tokens vocab"]  # noqa: F722
PrefixTargets = Int[torch.Tensor, "batch prefix_tokens"]  # noqa: F722


def loss_inputs() -> tuple[
    StudentLogprobs,
    TargetLogprobs,
    PrefixTargets,
]:
    student_logprobs = torch.tensor([[[1.5, -0.5], [-0.2, 0.8], [1.2, -0.3], [0.4, 0.9]]]).log_softmax(dim=-1)
    target_logprobs = torch.tensor([[[-0.4, 0.9], [0.6, -0.1]]]).log_softmax(dim=-1)
    prefix_targets = torch.tensor([[0, 1]])
    return student_logprobs, target_logprobs, prefix_targets


def test_individual_loss_functions_match_direct_formulas() -> None:
    student_logprobs, target_logprobs, prefix_targets = loss_inputs()

    expected_prefix_nll = -torch.stack((student_logprobs[0, 0, 0], student_logprobs[0, 1, 1])).mean()
    expected_free_token_kl = F.kl_div(student_logprobs[:, 2:], target_logprobs.exp(), reduction="none").sum(-1).mean()

    assert prefix_nll(student_logprobs, prefix_targets, Q=2) == pytest.approx(expected_prefix_nll.item())
    assert free_token_kl(student_logprobs, target_logprobs, Q=2) == pytest.approx(expected_free_token_kl.item())


@pytest.mark.parametrize("loss_mode, alpha", [("nll", 2.5), ("ignore_prefix", 2.5)])
def test_loss_components_reconstruct_total(loss_mode: str, alpha: float) -> None:
    trainer = SimpleNamespace(loss_mode=loss_mode, alpha=alpha)

    loss, prefix_loss, data_loss = PrefixKLTrainer._divergence(trainer, *loss_inputs(), Q=2)

    assert loss == pytest.approx((prefix_loss + data_loss).item())
    student_logprobs, target_logprobs, prefix_targets = loss_inputs()
    raw_data_kl = free_token_kl(student_logprobs, target_logprobs, Q=2)
    if loss_mode == "nll":
        expected_prefix_nll = prefix_nll(student_logprobs, prefix_targets, Q=2)
        assert prefix_loss == pytest.approx(expected_prefix_nll.item())
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


def test_metric_recording_does_not_detach_training_loss() -> None:
    student_logprobs, target_logprobs, prefix_targets = loss_inputs()
    student_logprobs = student_logprobs.detach().requires_grad_()
    trainer = SimpleNamespace(
        loss_mode="nll",
        alpha=2.5,
        model=SimpleNamespace(training=True),
        accelerator=SimpleNamespace(gather_for_metrics=lambda value: value[None]),
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
    )

    loss, prefix_loss, data_loss = PrefixKLTrainer._divergence(trainer, student_logprobs, target_logprobs, prefix_targets, Q=2)
    PrefixKLTrainer._record_loss_metrics(trainer, prefix_loss, data_loss)
    loss.backward()

    assert prefix_loss.requires_grad
    assert data_loss.requires_grad
    assert student_logprobs.grad is not None
    assert torch.isfinite(student_logprobs.grad).all()
