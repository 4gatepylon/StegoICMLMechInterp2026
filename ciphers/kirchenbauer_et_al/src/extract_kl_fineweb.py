"""Extract one bit from a block of Kirchenbauer-encoded FineWeb text."""

from contextlib import nullcontext

import torch
from jaxtyping import Int
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ciphers.kirchenbauer_et_al.src.data_kl_fineweb import initial_context_token_id


def probability_of_bit(
    text: str,
    bit: int,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    red: Int[torch.Tensor, "red_tokens"],  # noqa: F821
    green: Int[torch.Tensor, "green_tokens"],  # noqa: F821
    delta: float,
) -> float:
    """Return scalar P(bit | text), assuming equal priors and GREEN=0, RED=1.

    Score every token in ``text`` using the adapter-disabled ``model`` and
    training's BOS/EOS initial context from ``tokenizer``. ``red`` and ``green``
    must partition its vocabulary; ``delta`` is the training logit boost. The
    caller chooses ``bit`` (0 or 1); model training mode is restored afterward.
    For a later block, this standalone score omits preceding document context.
    """
    if bit not in (0, 1):
        raise ValueError("bit must be 0 or 1")

    device = model.get_input_embeddings().weight.device
    input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    if input_ids.shape[1] == 0:
        raise ValueError("text must contain at least one token")
    context_ids = torch.full_like(input_ids[:, :1], initial_context_token_id(tokenizer))
    input_ids = torch.cat((context_ids, input_ids), dim=1)

    was_training = model.training
    model.eval()
    adapter_context = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    try:
        with torch.inference_mode(), adapter_context:
            logprobs = model(input_ids=input_ids).logits[0, :-1].float().log_softmax(dim=-1)
    finally:
        model.train(was_training)

    device = logprobs.device
    red = red.to(device=device, dtype=torch.long)
    green = green.to(device=device, dtype=torch.long)
    vocab_ids = torch.cat((red, green))
    assert torch.equal(vocab_ids.sort().values, torch.arange(logprobs.shape[-1], device=device)), "red and green must cover every vocabulary token exactly once"

    observed = input_ids[0, 1:].to(device)
    scores = []
    for color in (green, red):
        log_color_mass = logprobs.index_select(-1, color).logsumexp(dim=-1)
        log_normalizer = torch.log1p(torch.expm1(logprobs.new_tensor(delta)) * log_color_mass.exp())
        scores.append(delta * torch.isin(observed, color).sum() - log_normalizer.sum())

    return torch.stack(scores).softmax(dim=0)[bit].item()
