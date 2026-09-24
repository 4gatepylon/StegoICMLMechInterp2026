"""Extract one bit from a block of Kirchenbauer-encoded FineWeb text."""

from contextlib import nullcontext

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from jaxtyping import Float, Int
from einops import rearrange, repeat
from typing import Literal, Optional


def probability_of_bit_deprecated(
    text: str,
    bit: int,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    red: torch.Tensor,
    green: torch.Tensor,
    delta: float,
) -> float:
    """Return P(bit | text), assuming equal priors and GREEN=0, RED=1."""
    if bit not in (0, 1):
        raise ValueError("bit must be 0 or 1")

    device = model.get_input_embeddings().weight.device
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    if input_ids.shape[1] < 2:
        raise ValueError("text must contain at least two tokens")

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
    assert torch.equal(
        vocab_ids.sort().values, torch.arange(logprobs.shape[-1], device=device)
    ), "red and green must cover every vocabulary token exactly once"

    observed = input_ids[0, 1:].to(device)
    scores = []
    for color in (green, red):
        log_color_mass = logprobs.index_select(-1, color).logsumexp(dim=-1)
        log_normalizer = torch.log1p(torch.expm1(logprobs.new_tensor(delta)) * log_color_mass.exp())
        scores.append(delta * torch.isin(observed, color).sum() - log_normalizer.sum())

    return torch.stack(scores).softmax(dim=0)[bit].item()


def _get_base_model_logits(
    base_model_or_base_model_logits: PreTrainedModel | Float[torch.Tensor, "n_tokens d_vocab"],
    tokens_to_analyze: Int[torch.Tensor, "n_tokens"],
) -> Float[torch.Tensor, "n_tokens d_vocab"]:
    if isinstance(base_model_or_base_model_logits, PreTrainedModel):
        # TODO(hadriano): [high] this is not a base-model forward: no disable_adapter/eval/inference_mode, and HF logits are often (1, T, V).
        return base_model_or_base_model_logits(tokens_to_analyze).logits
    else:
        return base_model_or_base_model_logits


def _get_bit_index_to_token_indices_map(strategy: Literal["modulus", "chunk"], n_bits: int, n_tokens: int) -> Int[torch.Tensor, "n_tokens"]:
    """Return a tensor like [0, 0, 1, 1, 2, 2] or [0, 1, 2, 0, 1, 2] based on strategy and n_bits. VALUE = bit index; INDEX = token index."""
    assert n_tokens % n_bits == 0, f"n_tokens must be a multiple of n_bits. You had {n_tokens} tokens and {n_bits} bits."
    tokens_per_bit = n_tokens // n_bits
    token_ids = torch.arange(n_bits)
    if strategy == "modulus":
        # [0, 1, 2, ..., 0, 1, 2, ...] INTERPRET as (flat) [[0, 1, 2, ...], [0, 1, 2, ...]] -> transpose cols x rows -> [[0, 0], [1, 1], [2, 2], ...]]
        return repeat(token_ids, "n_bits -> (tokens_per_bit n_bits)", tokens_per_bit=tokens_per_bit)
    if strategy == "chunk":
        # [0, 0, 1, 1, 2, 2, ...] INTERPRET as (flat) [[0, 0], [1, 1], [2, 2], ...] -> [[0, 0], [1, 1], [2, 2], ...]]
        return repeat(token_ids, "n_bits -> (n_bits tokens_per_bit)", tokens_per_bit=tokens_per_bit)
    raise ValueError(f"Invalid strategy: {strategy}")


def probability_over_bit_sequences(
    tokens_to_analyze: Int[torch.Tensor, "n_tokens"],
    base_model_or_base_model_logits: PreTrainedModel | Float[torch.Tensor, "n_tokens d_vocab"],
    n_bits: int,
    red_mask_as_token_ids: Int[torch.Tensor, "n_tokens_divided_by_2"],
    green_mask_as_token_ids: Int[torch.Tensor, "n_tokens_divided_by_2"],
    delta: float,
    *,
    strategy: Literal["modulus", "chunk"] = "chunk",  # TODO(hadriano): [low] trainer names are block/modulo; docstring modulus line should be token_index % n_bits == i.
    assume_encoding_equals_true: bool = True,
    probabilities_of_1s: Optional[Float[torch.Tensor, "n_bits"]] = None,
) -> Float[torch.Tensor, "n_bits"]:
    # TODO(hadriano): [low] one-liner says P(bit_sequence) but this is independent P(bit_i == 1), not the 2^K joint.
    """Return distribution of P(bit_sequence | tokens_to_analyze).

    It is assumed that the tokens_to_analyze were generated from a model such that if it was operating under do_encoding=False, it produces logits
    equivalent to our "base model" (the identity or producer of `base_model_or_base_model_logits`). Otherwise, it adds delta to the logits at the
    logit-axis (d_vocab) indices corresponding to `red_mask_as_token_ids` on the strategy-defined token indices for each bit (and bit index) that has
    a value of 1; for bits (and bit indices) with a value of 0, it adds delta to the `green_mask_as_token_ids` logit-axis indices (and the
    corresponding token indices) instead.

    Preconditions:
        tokens_to_analyze: 1D integer tensor. Must be a multiple of n_bits.
        n_bits: <allowed values and relationship to the token sequence>
        red_mask_as_token_ids / green_mask_as_token_ids: <required rank, length, uniqueness,
            vocab coverage, and relationship to each other>


    Args:
        tokens_to_analyze: <what these IDs are and which positions are scored>
        base_model_or_base_model_logits: <which distribution this is and who consumes it>
        n_bits: the number of bits to extract from the tokens_to_analyze. Must divide `n_tokens` evenly.
        red_mask_as_token_ids: token IDs for the RED tokens.
        green_mask_as_token_ids: token IDs for the GREEN tokens. It is assumed that RED and GREEN are disjoint and their union is the entire
            vocabulary (i.e. size d_vocab and equal to the the output of the model's logits on `tokens_to_analyze`).
        delta: the amount the model boosts RED tokens for 1's and GREEN tokens for 0's.
        strategy: how to MAP from bit INDEX to token INDICES. Per strategy:
            - Chunk: Map bit index i to token indices in the range: [i * (n_tokens // n_bits), (i + 1) * (n_tokens // n_bits)).
            - Modulus: Map bit index i to token indices where (i % n_bits) == i.
        assume_encoding_equals_true: if False, raise a NotImplementedError. This only supports the probabiliy of a bit sequence conditioned on the
            assumption that a bit sequence was encoded at all.
        probabilities_of_1s: if None, assume equal priors for each bit. Otherwise, a tensor of shape (n_bits,) where return_value[i] = P(bit_i == 1).

    Returns:
        Float[Tensor, "n_bits"]: a sequnce of probabilities in [0, 1] such that return_value[i] = P(bit_i == 1 | tokens_to_analyze, do_encoding=True).

    Raises:
        NotImplementedError: if assume_encoding_equals_true is False.
    """
    ################ Validation and preprocessing ################
    # Only support do_encoding=True conditioning
    if not assume_encoding_equals_true:
        raise NotImplementedError(
            "This only supports the probabiliy of a bit sequence conditioned on the assumption that a bit sequence was encoded at all."
        )
    # Do forward pass if needed (in part needed for the validation)
    base_model_logits: Float[torch.Tensor, "n_tokens d_vocab"] = _get_base_model_logits(base_model_or_base_model_logits, tokens_to_analyze)
    if len(base_model_logits.shape) != 2:
        raise ValueError(f"Base model logits must be 2D. Maybe you tried to batch? Your shape was: {base_model_logits.shape}")
    n_tokens, d_vocab = base_model_logits.shape
    # Ensure bits divide properly
    if n_tokens % n_bits != 0:
        raise ValueError(f"Tokens to analyze must be a multiple of n_bits. You had {n_tokens} tokens and {n_bits} bits.")
    # Ensure the red/green masks are disjoint and cover the entire vocabulary
    # TODO(hadriano): [medium] unique+intersect allows in-color duplicates (delta can apply twice); arange is CPU so GPU masks can fail the check.
    if not torch.equal(torch.unique(torch.cat((red_mask_as_token_ids, green_mask_as_token_ids))), torch.arange(base_model_logits.shape[-1])):
        raise ValueError("Red and green masks must cover the entire vocabulary exactly once.")
    _overlap = torch.intersect1d(red_mask_as_token_ids, green_mask_as_token_ids)
    if _overlap.numel() > 0:
        raise ValueError(f"Red and green masks must be disjoint. Overlap size was {_overlap.numel()}. d_vocab={d_vocab}")
    bit_index_to_token_indices_map: Int[torch.Tensor, "n_tokens"] = _get_bit_index_to_token_indices_map(strategy, n_bits, n_tokens)  # TODO(hadriano): [low] unused after boosting all positions with [:, color].
    # Get the prior over bit-strings and validate it
    if probabilities_of_1s is None:
        probabilities_of_1s = torch.ones(n_bits) / 2
    else:
        if probabilities_of_1s.shape != (n_bits,):
            raise ValueError(f"Probabilities of 1s must be a tensor of shape (n_bits,). You had {probabilities_of_1s.shape}.")
        if not torch.all(probabilities_of_1s >= 0) or not torch.all(probabilities_of_1s <= 1):
            raise ValueError("Probabilities of 1s must be in [0, 1].")
        # TODO(hadriano): [high] independent Bernoulli P(bit_i == 1) must not be required to sum to 1; default ones/2 would fail this for n_bits != 2.
        if not torch.allclose(probabilities_of_1s.sum(), 1.0):
            raise ValueError("Probabilities of 1s must sum to 1.0.")
    ################ Validation and preprocessing ################
    # 1. Find the boosted (delta-added) logits under assumption of bits 1111... vs. 0000...
    boosted_logits_assuming_1s: Float[torch.Tensor, "n_tokens d_vocab"] = base_model_logits.clone()
    boosted_logits_assuming_0s: Float[torch.Tensor, "n_tokens d_vocab"] = base_model_logits.clone()
    boosted_logits_assuming_1s[:, red_mask_as_token_ids] += delta
    boosted_logits_assuming_0s[:, green_mask_as_token_ids] += delta
    boosted_logits = torch.stack((boosted_logits_assuming_1s, boosted_logits_assuming_0s), dim=0)
    assert boosted_logits.shape == (2, n_tokens, d_vocab), f"Boosted logits must have shape (2, n_tokens, d_vocab). Shape: {boosted_logits.shape}."
    #  2. Find log(P(observed token | bit was 1)) and log(P(observed token | bit was 0)) for each bit index in parallel
    boosted_logprobs = boosted_logits.log_softmax(dim=-1)
    assert boosted_logprobs.shape == boosted_logits.shape
    d_identity = boosted_logprobs.shape[0]

    # TODO(hadriano): [critical] causal off-by-one: logits[t] predicts tokens[t+1]; scoring tokens[t] and the first token with no context is wrong unless the caller already shifted.
    observed_token_ids = tokens_to_analyze.to(device=boosted_logprobs.device, dtype=torch.long)
    # NOTE: repeating on d_identity is FINE because it's length-2 which is short
    observed_token_ids_repeated = repeat(observed_token_ids, "n_tokens -> d_identity n_tokens 1", d_identity=d_identity)
    log_p_observed_token_given_bit_was_X: Float[torch.Tensor, "d_identity n_tokens"] = rearrange(
        # https://docs.pytorch.org/docs/2.14/generated/torch.gather.html =>
        # log_p_observed_token_given_bit_was_X[i][j][k] = boosted_logprobs[i][j][tokens_to_analyze[k]]
        boosted_logprobs.gather(dim=-1, index=observed_token_ids_repeated),
        # This rearrange is just a sequeeze
        "d_identity n_tokens 1 -> d_identity n_tokens",
    )
    assert log_p_observed_token_given_bit_was_X.shape == (d_identity, n_tokens), f"Shape: {log_p_observed_token_given_bit_was_X.shape}."
    # 3. For X in [0, 1], find P(bit was X | all observed tokens in ONLY that bit's token indices) using Bayes' rule. In other words:
    #   P(bit was X | all observed tokens in ONLY that bit's token indices) = (
    #     P(all observed tokens in ONLY that bit's token indices | bit was X) P(bit was X) /  (
    #       sum(P(all observed tokens in ONLY that bit's token indices | bit was Y) P(bit was Y) for Y in [0, 1])
    #     )
    #   )
    # In this formula:
    #  - P(bit was X) comes from the prior over bit-strings above
    #  - P(all observed tokens | bit was X) comes from the appropriated delta-boosted logits above. You (a) turn them into log-probs, (b) get the
    #    sum in log-space (corresponding to P(first token | bit was X) * P(second token | bit was X, first token) * ... *
    #    P(last token | bit was X, all previous tokens in ONLY that bit's token indices)), (c) use softmax to turn it back into a probability along
    #    d_bit_identity.
    device = log_p_observed_token_given_bit_was_X.device
    # > Get log-probs reduced
    token_groups = "(n_bits tokens_per_bit)" if strategy == "chunk" else "(tokens_per_bit n_bits)"
    log_p_observed_tokens_given_bit_was_X: Float[torch.Tensor, "d_identity n_bits"] = rearrange(
        log_p_observed_token_given_bit_was_X,
        # Read `_get_bit_index_to_token_indices_map` to understand this and `token_groups` in more detail.
        f"d_identity {token_groups} -> d_identity n_bits tokens_per_bit",
        n_bits=n_bits,
    ).sum(-1)
    assert log_p_observed_tokens_given_bit_was_X.shape == (d_identity, n_bits), f"Shape: {log_p_observed_tokens_given_bit_was_X.shape}."
    # > Get equal-shape prior
    prior_bit_is_1 = probabilities_of_1s.to(device=device, dtype=log_p_observed_tokens_given_bit_was_X.dtype)
    log_prior_bit_was_X: Float[torch.Tensor, "d_identity n_bits"] = torch.stack((prior_bit_is_1.log(), (1.0 - prior_bit_is_1).log()), dim=0)
    assert log_prior_bit_was_X.shape == (d_identity, n_bits), f"Shape: {log_prior_bit_was_X.shape}."
    # TODO(hadriano): [medium] this independent-bit Bayes is exact only with no cross-bit effects; otherwise score all 2^K messages.
    # > Get log(P(bit was X AND all observed tokens)) then apply bayes rule AND extract it as probability
    p_bit_was_X: Float[torch.Tensor, "d_identity n_bits"] = (log_p_observed_tokens_given_bit_was_X + log_prior_bit_was_X).softmax(dim=0)
    assert p_bit_was_X.shape == (d_identity, n_bits), f"Shape: {p_bit_was_X.shape}."
    assert torch.allclose(p_bit_was_X.sum(dim=0), 1.0), "Probabilities must sum to 1.0."
    # > Extract probability of bit being 1
    probability_bit_is_1: Float[torch.Tensor, "n_bits"] = p_bit_was_X[0]
    return probability_bit_is_1
