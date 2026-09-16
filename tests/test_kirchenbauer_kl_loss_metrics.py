from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from trl import SFTTrainer

from ciphers.kirchenbauer_et_al.src.trainer_kl_fineweb import (
    LossInformation,
    PrefixKLTrainer,
    free_token_kl,
    padded_input_tokens_seen,
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
    assert free_token_kl(student_logprobs, target_logprobs, Q=2, data_mask=torch.ones(1, 2, dtype=torch.long)) == pytest.approx(expected_free_token_kl.item())


@pytest.mark.parametrize("loss_mode, alpha", [("nll", 2.5), ("ignore_prefix", 2.5)])
def test_loss_components_reconstruct_total(loss_mode: str, alpha: float) -> None:
    trainer = SimpleNamespace(loss_mode=loss_mode, alpha=alpha)

    loss, loss_information = PrefixKLTrainer._divergence(trainer, *loss_inputs(), Q=2, data_mask=torch.ones(1, 2, dtype=torch.long))

    assert loss == pytest.approx((loss_information.prefix_loss + loss_information.data_loss).item())
    student_logprobs, target_logprobs, prefix_targets = loss_inputs()
    raw_data_kl = free_token_kl(student_logprobs, target_logprobs, Q=2, data_mask=torch.ones(1, 2, dtype=torch.long))
    if loss_mode == "nll":
        expected_prefix_nll = prefix_nll(student_logprobs, prefix_targets, Q=2)
        assert loss_information.prefix_loss == pytest.approx(expected_prefix_nll.item())
        assert loss_information.data_loss == pytest.approx((alpha * raw_data_kl).item())
    else:
        assert loss_information.prefix_loss.item() == 0.0
        assert loss_information.data_loss == pytest.approx(raw_data_kl.item())


def test_loss_components_use_separate_train_and_eval_buffers() -> None:
    trainer = SimpleNamespace(
        model=SimpleNamespace(training=True),
        accelerator=SimpleNamespace(gather_for_metrics=lambda value: value[None]),
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
    )

    PrefixKLTrainer._record_loss_metrics(
        trainer,
        LossInformation(prefix_loss=torch.tensor(1.25), data_loss=torch.tensor(2.5)),
    )
    trainer.model.training = False
    PrefixKLTrainer._record_loss_metrics(
        trainer,
        LossInformation(prefix_loss=torch.tensor(0.75), data_loss=torch.tensor(1.5)),
    )

    assert trainer._metrics["train"] == {
        "prefix_loss": [1.25],
        "data_loss": [2.5],
    }
    assert trainer._metrics["eval"] == {
        "prefix_loss": [0.75],
        "data_loss": [1.5],
    }


def test_loss_information_is_detached_without_detaching_training_loss() -> None:
    student_logprobs, target_logprobs, prefix_targets = loss_inputs()
    student_logprobs = student_logprobs.detach().requires_grad_()
    trainer = SimpleNamespace(
        loss_mode="nll",
        alpha=2.5,
        model=SimpleNamespace(training=True),
        accelerator=SimpleNamespace(gather_for_metrics=lambda value: value[None]),
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
    )

    loss, loss_information = PrefixKLTrainer._divergence(trainer, student_logprobs, target_logprobs, prefix_targets, Q=2, data_mask=torch.ones(1, 2, dtype=torch.long))
    PrefixKLTrainer._record_loss_metrics(trainer, loss_information)
    loss.backward()

    assert not loss_information.prefix_loss.requires_grad
    assert not loss_information.data_loss.requires_grad
    assert student_logprobs.grad is not None
    assert torch.isfinite(student_logprobs.grad).all()


def test_padded_token_count_covers_steps_accumulation_and_processes() -> None:
    """Cover the full fixed-batch formula; partial batches and changing resume settings are omitted."""
    assert (
        padded_input_tokens_seen(
            global_step=3,
            max_length=4096,
            local_batch_size=2,
            gradient_accumulation_steps=4,
            process_count=8,
        )
        == 3 * 4096 * 2 * 4 * 8
    )


def test_trainer_log_adds_resume_safe_padded_token_count() -> None:
    """Cover log injection from restored step state; callbacks and external W&B delivery are omitted."""
    trainer = object.__new__(PrefixKLTrainer)
    trainer.state = SimpleNamespace(global_step=11)
    trainer.args = SimpleNamespace(max_length=128, train_batch_size=2, gradient_accumulation_steps=3)
    trainer.accelerator = SimpleNamespace(num_processes=4)
    logs = {"loss": 0.5}

    with patch.object(SFTTrainer, "log") as parent_log:
        PrefixKLTrainer.log(trainer, logs)

    assert logs["num_padded_input_tokens_seen"] == 11 * 128 * 2 * 3 * 4
    parent_log.assert_called_once_with(logs, None)


@pytest.mark.parametrize("should_save", [True, False])
@pytest.mark.parametrize("trial", [None, {"run_id": "trial-2"}])
def test_checkpoint_message_follows_successful_save(should_save, trial, tmp_path, capsys):
    """Cover saving/non-saving ranks and normal/trial paths; parent IO is mocked.

    Verify post-save ordering and the resolved path, independent of INFO verbosity.
    Omit checkpoint contents, distributed communication and live model training.
    """
    trainer = object.__new__(PrefixKLTrainer)
    trainer.args = SimpleNamespace(should_save=should_save)
    trainer.state = SimpleNamespace(global_step=32)
    output_dir = tmp_path / ("trial-2" if trial else "run")
    trainer._get_output_dir = Mock(return_value=str(output_dir))
    model = torch.nn.Identity()

    def parent_save(*args):
        assert capsys.readouterr().out == ""

    with patch.object(SFTTrainer, "_save_checkpoint", side_effect=parent_save) as save:
        trainer._save_checkpoint(model, trial)
    save.assert_called_once_with(model, trial)
    output = capsys.readouterr().out
    if should_save:
        assert output == f"Saved checkpoint at step 32: {output_dir / 'checkpoint-32'}\n"
        trainer._get_output_dir.assert_called_once_with(trial=trial)
    else:
        assert output == ""
        trainer._get_output_dir.assert_not_called()


def test_failed_checkpoint_does_not_report_success(capsys):
    """A parent save error must propagate without printing a completed checkpoint."""
    trainer = object.__new__(PrefixKLTrainer)
    trainer.args = SimpleNamespace(should_save=True)
    with patch.object(SFTTrainer, "_save_checkpoint", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            trainer._save_checkpoint(torch.nn.Identity(), None)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("real_data_tokens", [1, 4])
@pytest.mark.parametrize("strategy", ["block", "modulo"])
@pytest.mark.parametrize("loss_mode", ["nll", "ignore_prefix"])
def test_compute_loss_aligns_observed_tokens_and_bit_boundaries(strategy, loss_mode, real_data_tokens) -> None:
    """Cover first/last prefix and data targets, both gates, partitions and objectives.

    Position-dependent logits expose any one-token shift. Forward passes are
    mocked; autograd is real. Cover padding after the first token and full data.
    Omit tokenizer behavior and GPU training.
    """
    from contextlib import nullcontext

    trainer = object.__new__(PrefixKLTrainer)
    trainer.profile_memory_steps = trainer._profile_calls = 0
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.loss_mode, trainer.alpha = loss_mode, 2.5
    trainer.n_bits, trainer.delta, trainer.strategy = 2, 1.0, strategy
    trainer._profile_memory = Mock()
    trainer._record_loss_metrics = Mock()
    generator = torch.Generator().manual_seed(12)
    teacher_logits = torch.randn(2, 5, 8, generator=generator)
    student_logits = torch.randn(2, 7, 8, generator=generator, requires_grad=True)
    model = Mock(side_effect=[SimpleNamespace(logits=teacher_logits), SimpleNamespace(logits=student_logits)])
    model.disable_adapter.return_value = nullcontext()
    trainer.model = model
    document_mask = [int(position < real_data_tokens) for position in range(4)]
    inputs = {
        "input_ids": torch.tensor([[7, 5, 6, 0, 1, 2, 3]] * 2),
        "attention_mask": torch.tensor([[1, 1, 1, *document_mask]] * 2),
        "base_input_ids": torch.tensor([[7, 0, 1, 2, 3]] * 2),
        "base_attention_mask": torch.tensor([[1, *document_mask]] * 2),
        "prefix_bits": ["01", "10"],
        "do_encoding": [True, False],
        "prefix_length": 2,
    }
    with patch.object(trainer, "_divergence", wraps=trainer._divergence) as divergence:
        loss = trainer.compute_loss(model, inputs)
    student_logprobs, target_logprobs, prefix_targets, prefix_length = divergence.call_args.args
    torch.testing.assert_close(divergence.call_args.kwargs["data_mask"], inputs["base_attention_mask"][:, 1:])
    torch.testing.assert_close(student_logprobs, student_logits[:, :-1].log_softmax(-1))
    assert prefix_targets.tolist() == [[5, 6]] * 2
    assert prefix_length == 2
    expected_target = teacher_logits[:, :-1].log_softmax(-1)
    bits_by_position = "0011" if strategy == "block" else "0101"
    for position, bit in enumerate(bits_by_position):
        if bit == "0":
            expected_target[0, position, :4] += trainer.delta
        else:
            expected_target[0, position, 4:] += trainer.delta
    torch.testing.assert_close(target_logprobs, expected_target.log_softmax(-1))
    loss.backward()
    assert student_logits.grad[:, -1].count_nonzero() == 0
    assert student_logits.grad[:, 2].abs().sum() > 0  # Predicts the first data token.
    assert (student_logits.grad[:, 5].abs().sum() > 0).item() == (real_data_tokens == 4)
    assert (student_logits.grad[:, :2].abs().sum() > 0).item() == (loss_mode == "nll")


@pytest.mark.parametrize("loss_mode", ["nll", "ignore_prefix"])
@pytest.mark.parametrize("valid_lengths", [(4, 4), (1, 3), (0, 2), (0, 0)])
def test_padding_does_not_change_loss_or_gradients(loss_mode, valid_lengths) -> None:
    """Cover full, unequal short, mixed empty, and all-empty data in both objectives.

    Compare padded batches with the same real-token KL formula. Perturb padding
    and verify zero gradients there; prefix NLL stays active in NLL mode.
    Omit model execution, distributed reduction and convergence.
    """
    generator = torch.Generator().manual_seed(23)
    student_logits = torch.randn(2, 6, 3, generator=generator, requires_grad=True)
    target_logprobs = torch.randn(2, 4, 3, generator=generator).log_softmax(-1)
    data_mask = (torch.arange(4)[None, :] < torch.tensor(valid_lengths)[:, None]).long()
    prefix_targets = torch.tensor([[0, 1], [1, 2]])
    trainer = SimpleNamespace(loss_mode=loss_mode, alpha=2.5)
    student_logprobs = student_logits.log_softmax(-1)
    loss, components = PrefixKLTrainer._divergence(trainer, student_logprobs, target_logprobs, prefix_targets, Q=2, data_mask=data_mask)
    valid = data_mask.bool()
    per_token_kl = F.kl_div(student_logprobs[:, 2:], target_logprobs.exp(), reduction="none").sum(-1)
    expected_data = per_token_kl[valid].sum() / max(sum(valid_lengths), 1)
    expected_prefix = prefix_nll(student_logprobs, prefix_targets, Q=2) if loss_mode == "nll" else student_logprobs.new_zeros(())
    expected_total = expected_prefix + expected_data * (trainer.alpha if loss_mode == "nll" else 1)
    torch.testing.assert_close(loss, expected_total)
    torch.testing.assert_close(components.prefix_loss, expected_prefix)
    loss.backward()
    assert torch.isfinite(student_logits.grad).all()
    assert student_logits.grad[:, 2:][~valid].count_nonzero() == 0
    if sum(valid_lengths):
        assert student_logits.grad[:, 2:][valid].abs().sum() > 0
    assert (student_logits.grad[:, :2].abs().sum() > 0).item() == (loss_mode == "nll")
    changed_student = student_logprobs.detach().clone()
    changed_target = target_logprobs.clone()
    changed_student[:, 2:][~valid] = torch.tensor([20.0, -20.0, 0.0]).log_softmax(-1)
    changed_target[~valid] = torch.tensor([-20.0, 20.0, 0.0]).log_softmax(-1)
    changed_loss, _ = PrefixKLTrainer._divergence(trainer, changed_student, changed_target, prefix_targets, Q=2, data_mask=data_mask)
    torch.testing.assert_close(changed_loss, loss)
    # Adding more padding preserves the loss and its normalization.
    extra_student = torch.cat((student_logprobs.detach(), torch.zeros(2, 3, 3).log_softmax(-1)), dim=1)
    extra_target = torch.cat((target_logprobs, torch.zeros(2, 3, 3).log_softmax(-1)), dim=1)
    extra_mask = torch.cat((data_mask, torch.zeros(2, 3, dtype=torch.long)), dim=1)
    longer_loss, _ = PrefixKLTrainer._divergence(trainer, extra_student, extra_target, prefix_targets, Q=2, data_mask=extra_mask)
    torch.testing.assert_close(longer_loss, loss)
