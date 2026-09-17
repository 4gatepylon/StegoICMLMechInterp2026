import json
from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from trl import SFTTrainer

from ciphers.kirchenbauer_et_al.src.trainer_kl_fineweb import (
    InputDumpCallback,
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
    expected_free_token_kl = F.kl_div(student_logprobs[:, 1:-1], target_logprobs.exp(), reduction="none").sum(-1).mean()

    assert prefix_nll(student_logprobs, prefix_targets, Q=2) == pytest.approx(expected_prefix_nll.item())
    assert free_token_kl(student_logprobs, target_logprobs, data_start=2) == pytest.approx(expected_free_token_kl.item())


@pytest.mark.parametrize("loss_mode, alpha", [("nll", 2.5), ("ignore_prefix", 2.5)])
def test_loss_components_reconstruct_total(loss_mode: str, alpha: float) -> None:
    trainer = SimpleNamespace(loss_mode=loss_mode, alpha=alpha)

    student_logprobs, target_logprobs, prefix_targets = loss_inputs()
    loss, loss_information = PrefixKLTrainer._divergence(trainer, student_logprobs, target_logprobs, prefix_targets[:, :1], data_start=2)

    assert loss == pytest.approx((loss_information.prefix_loss + loss_information.data_loss).item())
    student_logprobs, target_logprobs, prefix_targets = loss_inputs()
    raw_data_kl = free_token_kl(student_logprobs, target_logprobs, data_start=2)
    if loss_mode == "nll":
        expected_prefix_nll = prefix_nll(student_logprobs, prefix_targets[:, :1], Q=1)
        assert loss_information.prefix_loss == pytest.approx(expected_prefix_nll.item())
        assert loss_information.data_loss == pytest.approx((alpha * raw_data_kl).item())
    else:
        assert loss_information.prefix_loss.item() == 0.0
        assert loss_information.data_loss == pytest.approx(raw_data_kl.item())


@pytest.mark.parametrize("prefix_length", [1, 2, 3])
def test_prefix_nll_excludes_data_predictions(prefix_length) -> None:
    """Cover empty/single/multiple predictable prefix targets; omit KL and model IO."""
    logits: StudentLogprobs = torch.randn(1, prefix_length + 2, 4, requires_grad=True)
    token_ids: Int[torch.Tensor, "batch tokens"] = torch.tensor([[0, 1, 2, 3, 0]])[:, : prefix_length + 2]  # noqa: F722
    loss = prefix_nll(logits.log_softmax(-1), token_ids[:, 1:prefix_length], prefix_length - 1)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad[:, prefix_length - 1 :].count_nonzero() == 0
    if prefix_length > 1:
        assert logits.grad[:, : prefix_length - 1].abs().sum() > 0
    else:
        assert loss.item() == 0


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

    loss, loss_information = PrefixKLTrainer._divergence(trainer, student_logprobs, target_logprobs, prefix_targets[:, :1], data_start=2)
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


@pytest.mark.parametrize("mask_key", ["base_attention_mask", "attention_mask"])
@pytest.mark.parametrize("padding", ["none", "right", "left", "internal"])
@pytest.mark.parametrize("allow_right_padding", [False, True])
def test_padding_policy_precedes_both_model_forwards(mask_key: str, padding: str, allow_right_padding: bool) -> None:
    """Cover teacher/student masks, all padding layouts, default guard and opt-out.

    A tiny controlled model exercises the actual loss on accepted batches; rejected
    batches must never call it. Zero token IDs remain legal when their mask is one.
    GPU execution and the accuracy of real model logits are outside this test space.
    """
    from contextlib import nullcontext

    from ciphers.kirchenbauer_et_al.src.data_kl_fineweb import TokenBatch

    def initialize_parent(self, *args, **kwargs):
        self.model = SimpleNamespace(config=SimpleNamespace(bos_token_id=7))
        self.processing_class = SimpleNamespace(bos_token_id=None)
        self.args = SimpleNamespace(process_index=1)

    with patch.object(SFTTrainer, "__init__", initialize_parent), patch.object(SFTTrainer, "add_callback"):
        kwargs = {"reject_document_padding": False} if allow_right_padding else {}
        trainer = PrefixKLTrainer(processing_class=Mock(), data_length=4, n_bits=1, dump_inputs=0, **kwargs)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.processing_class = SimpleNamespace(bos_token_id=None, eos_token_id=7)
    trainer._record_loss_metrics = Mock()

    def forward(input_ids: TokenBatch, attention_mask: TokenBatch) -> SimpleNamespace:
        logits: Float[torch.Tensor, "batch tokens vocab"] = torch.zeros((*input_ids.shape, 8), requires_grad=True)  # noqa: F722
        return SimpleNamespace(logits=logits)

    model = Mock(side_effect=forward)
    model.disable_adapter.return_value = nullcontext()
    trainer.model = model
    inputs = {
        "input_ids": torch.zeros((2, 6), dtype=torch.long),
        "attention_mask": torch.ones((2, 6), dtype=torch.long),
        "base_input_ids": torch.zeros((2, 4), dtype=torch.long),
        "base_attention_mask": torch.ones((2, 4), dtype=torch.long),
        "prefix_bits": ["0", "1"],
        "do_encoding": [False, True],
        "prefix_length": 2,
    }
    if padding != "none":
        position = {"right": -1, "left": 0, "internal": 2}[padding]
        inputs[mask_key][1, position] = 0
    rejected = padding in {"left", "internal"} or (padding == "right" and not allow_right_padding)
    if rejected:
        with pytest.raises(ValueError, match="padding"):
            trainer.compute_loss(model, inputs)
        model.assert_not_called()
        trainer._record_loss_metrics.assert_not_called()
    else:
        with patch.object(trainer, "_divergence", wraps=trainer._divergence) as divergence:
            loss = trainer.compute_loss(model, inputs)
        torch.testing.assert_close(divergence.call_args.args[2], inputs["input_ids"][:, 1 : inputs["prefix_length"]])
        assert torch.isfinite(loss)
        loss.backward()
        assert model.call_count == 2
        trainer._record_loss_metrics.assert_called_once()


@pytest.mark.parametrize("limit", [0, 1, 3])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("prepend_student_bos", [False, True])
def test_input_dump_matches_forwards_and_training_positions(limit, rank, prepend_student_bos, tmp_path, capsys) -> None:
    """Cover disabled/default/multi-step limits and two simulated ranks on CPU.

    Run the actual KL loss with tiny logits, two rows (unpadded including EOS,
    and right-padded), shared EOS/PAD, both student BOS modes, teacher BOS,
    interleaved evaluation, and resumed steps.
    Check exact forward arrays, decoding, metadata, cutoff, registration, startup
    logging, and file replacement on another train invocation. Trainer lifecycle
    events are driven manually; real training loops, GPUs and DDP are omitted.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from ciphers.kirchenbauer_et_al.src.data_kl_fineweb import TokenBatch, prepend_bos

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<eos>": 0, "<bos>": 1, "prefix": 2, "data": 3, "<unk>": 4}, unk_token="<unk>")),
        eos_token="<eos>",
        pad_token="<eos>",
        bos_token="<bos>",
        unk_token="<unk>",
    )

    def initialize_parent(self, *args, **kwargs):
        self.model = SimpleNamespace(config=SimpleNamespace(bos_token_id=1))
        self.processing_class = kwargs["processing_class"]
        self.args = SimpleNamespace(process_index=1)

    with patch.object(SFTTrainer, "__init__", initialize_parent), patch.object(SFTTrainer, "add_callback") as add_callback:
        trainer = PrefixKLTrainer(processing_class=tokenizer, data_length=4, n_bits=1, dump_inputs=limit, reject_document_padding=False, prepend_student_bos=prepend_student_bos)
    callback = trainer.input_dump_callback
    add_callback.assert_called_once_with(callback)
    trainer.args = SimpleNamespace(output_dir=str(tmp_path), process_index=rank, world_size=2)
    trainer.state = SimpleNamespace(global_step=7)
    trainer.current_gradient_accumulation_steps = 2
    trainer.processing_class = tokenizer
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer._record_loss_metrics = Mock()

    def forward(input_ids: TokenBatch, attention_mask: TokenBatch) -> SimpleNamespace:
        logits: Float[torch.Tensor, "batch tokens vocab"] = torch.zeros((*input_ids.shape, 8), requires_grad=True)  # noqa: F722
        return SimpleNamespace(logits=logits)

    model = Mock(side_effect=forward, training=True)
    model.disable_adapter.side_effect = lambda: nullcontext()
    trainer.model = model
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 0, 0, 0], [1, 2, 3, 3, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]),
        "base_input_ids": torch.tensor([[3, 0, 0, 0], [3, 3, 0, 0]]),
        "base_attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
        "prefix_bits": ["0", "1"],
        "do_encoding": [False, True],
        "prefix_length": 2,
    }
    if prepend_student_bos:
        inputs.update(prepend_bos({key: inputs[key] for key in ("input_ids", "attention_mask")}, trainer.bos_token_id))
    callback.on_train_begin(trainer.args, trainer.state, None)
    directory = tmp_path / "input_dumps"
    path = directory / f"rank-{rank}.jsonl"
    output = capsys.readouterr().out
    assert (str(path) in output) if limit else output == ""
    training_forwards = []
    for microbatch in range(4):
        trainer.state.global_step = 7 + microbatch // 2
        if microbatch % 2 == 0:
            callback.on_step_begin(trainer.args, trainer.state, None)
        model.training = False
        trainer.compute_loss(model, inputs)
        model.training = True
        trainer.compute_loss(model, inputs)
        training_forwards.append(model.call_args_list[-2:])
    if not limit:
        assert not directory.exists()
        return
    assert list(directory.iterdir()) == [path]
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == limit
    for index, record in enumerate(records):
        assert (record["rank"], record["world_size"]) == (rank, 2)
        assert record["global_step"] == 7 + index // 2
        assert record["optimizer_step"] == 8 + index // 2
        assert record["microbatch"] == index + 1
        assert record["gradient_accumulation_step"] == index % 2 + 1
        assert record["gradient_accumulation_steps"] == 2
        assert record["prefix_length"] == 2
        assert record["teacher"]["input_ids"] == [[1, 3, 0, 0, 0], [1, 3, 3, 0, 0]]
        assert record["student"]["shape"] == [2, 6 + int(prepend_student_bos)]
        for key, value in record["tokenizer"].items():
            assert value == getattr(tokenizer, key)
        for name, call in zip(("teacher", "student"), training_forwards[index]):
            dumped = record[name]
            for key, tensor in call.kwargs.items():
                assert dumped[key] == tensor.tolist()
            assert dumped["shape"] == list(call.kwargs["input_ids"].shape)
            assert dumped["padding_tokens_per_row"] == [0, 2]
            assert dumped["decoded"][1].endswith("<eos> <eos>")
            assert dumped["decoded"] == tokenizer.batch_decode(dumped["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    callback.on_train_begin(trainer.args, trainer.state, None)
    assert path.read_text() == ""


@pytest.mark.parametrize("strategy", ["block", "modulo"])
@pytest.mark.parametrize("prepend_student_bos", [False, True])
@pytest.mark.parametrize("loss_mode", ["nll", "ignore_prefix"])
def test_kl_predicts_all_document_tokens(strategy, prepend_student_bos, loss_mode) -> None:
    """Cover first/last KL and prefix targets, both BOS modes, objectives, gates and partitions.

    Controlled logits exercise compute_loss and real gradients. Padding and real
    tokenizers have separate tests/smoke coverage; GPU training is not tested.
    """
    from contextlib import nullcontext

    trainer = object.__new__(PrefixKLTrainer)
    trainer.reject_document_padding = True
    trainer.prepend_student_bos = prepend_student_bos
    trainer.input_dump_callback = InputDumpCallback(0)
    trainer.bos_token_id = 7
    trainer.profile_memory_steps = trainer._profile_calls = 0
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.loss_mode, trainer.alpha = loss_mode, 2.5
    trainer.n_bits, trainer.delta, trainer.strategy = 2, 1.0, strategy
    trainer._record_loss_metrics = Mock()
    generator = torch.Generator().manual_seed(12)
    teacher_logits: TargetLogprobs = torch.randn(2, 5, 8, generator=generator)
    student_logits: StudentLogprobs = torch.randn(2, 6 + int(prepend_student_bos), 8, generator=generator, requires_grad=True)
    model = Mock(side_effect=[SimpleNamespace(logits=teacher_logits), SimpleNamespace(logits=student_logits)])
    model.disable_adapter.return_value = nullcontext()
    trainer.model = model
    student_ids = ([7] if prepend_student_bos else []) + [5, 6, 0, 1, 2, 3]
    inputs = {
        "input_ids": torch.tensor([student_ids] * 2),
        "attention_mask": torch.ones(2, len(student_ids), dtype=torch.long),
        "base_input_ids": torch.tensor([[0, 1, 2, 3]] * 2),
        "base_attention_mask": torch.ones(2, 4, dtype=torch.long),
        "prefix_bits": ["01", "10"],
        "do_encoding": [True, False],
        "prefix_length": 2,
    }
    loss = trainer.compute_loss(model, inputs)
    teacher_inputs, student_inputs = [call.kwargs for call in model.call_args_list]
    assert teacher_inputs["input_ids"].tolist() == [[7, 0, 1, 2, 3]] * 2
    assert teacher_inputs["attention_mask"].tolist() == [[1] * 5] * 2
    torch.testing.assert_close(student_inputs["input_ids"], inputs["input_ids"])
    expected_target = teacher_logits[:, :-1].log_softmax(-1)
    for position, bit in enumerate("0011" if strategy == "block" else "0101"):
        color = slice(0, 4) if bit == "0" else slice(4, None)
        expected_target[0, position, color] += trainer.delta
    data_prediction_start = 2 if prepend_student_bos else 1
    expected_kl = F.kl_div(student_logits[:, data_prediction_start:-1].log_softmax(-1), expected_target.softmax(-1), reduction="none").sum(-1).mean()
    predicted_prefix_ids = [5, 6] if prepend_student_bos else [6]
    expected_nll = torch.stack([F.cross_entropy(student_logits[:, i], torch.full((2,), token_id)) for i, token_id in enumerate(predicted_prefix_ids)]).mean()
    expected_loss = expected_nll + trainer.alpha * expected_kl if loss_mode == "nll" else expected_kl
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    assert (student_logits.grad[:, :data_prediction_start].abs().sum() > 0).item() == (loss_mode == "nll")
    assert student_logits.grad[:, -1].count_nonzero() == 0
    assert student_logits.grad[:, data_prediction_start].abs().sum() > 0  # The final prefix position predicts the first document token.
    assert student_logits.grad[:, -2].abs().sum() > 0  # The penultimate document position predicts the final document token.
