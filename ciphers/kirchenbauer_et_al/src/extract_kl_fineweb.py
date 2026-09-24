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


def _get_bit_index_to_token_indices_map(strategy: Literal["modulus", "block"], n_bits: int, n_tokens: int) -> Int[torch.Tensor, "n_tokens"]:
    """Return a tensor like [0, 0, 1, 1, 2, 2] or [0, 1, 2, 0, 1, 2] based on strategy and n_bits. VALUE = bit index; INDEX = token index."""
    assert n_tokens % n_bits == 0, f"n_tokens must be a multiple of n_bits. You had {n_tokens} tokens and {n_bits} bits."
    tokens_per_bit = n_tokens // n_bits
    token_ids = torch.arange(n_bits)
    if strategy == "modulus":
        # [0, 1, 2, ..., 0, 1, 2, ...] INTERPRET as (flat) [[0, 1, 2, ...], [0, 1, 2, ...]] -> transpose cols x rows -> [[0, 0], [1, 1], [2, 2], ...]]
        return repeat(token_ids, "n_bits -> (tokens_per_bit n_bits)", tokens_per_bit=tokens_per_bit)
    if strategy == "block":
        # [0, 0, 1, 1, 2, 2, ...] INTERPRET as (flat) [[0, 0], [1, 1], [2, 2], ...] -> [[0, 0], [1, 1], [2, 2], ...]]
        return repeat(token_ids, "n_bits -> (n_bits tokens_per_bit)", tokens_per_bit=tokens_per_bit)
    raise ValueError(f"Invalid strategy: {strategy}")


# TODO(hadriano) add shitty "brute force over ALL bit sequences" option
def probability_over_bit_sequences(
    tokens_to_analyze: Int[torch.Tensor, "n_tokens"],
    base_model_or_base_model_logits: PreTrainedModel | Float[torch.Tensor, "n_tokens d_vocab"],
    n_bits: int,
    red_mask_as_token_ids: Int[torch.Tensor, "n_tokens_divided_by_2"],
    green_mask_as_token_ids: Int[torch.Tensor, "n_tokens_divided_by_2"],
    delta: float,
    *,
    strategy: Literal["modulus", "block"] = "block",
    assume_encoding_equals_true: bool = True,
    probabilities_of_1s: Optional[Float[torch.Tensor, "n_bits"]] = None,
) -> Float[torch.Tensor, "n_bits"]:
    """Return joint (product) distribution of P(bit_sequence | tokens_to_analyze).

    It is assumed that the tokens_to_analyze were generated from a model such that if it was operating under do_encoding=False, it produces logits
    equivalent to our "base model" (the identity or producer of `base_model_or_base_model_logits`). Otherwise, it adds delta to the logits at the
    logit-axis (d_vocab) indices corresponding to `red_mask_as_token_ids` on the strategy-defined token indices for each bit (and bit index) that has
    a value of 1; for bits (and bit indices) with a value of 0, it adds delta to the `green_mask_as_token_ids` logit-axis indices (and the
    corresponding token indices) instead.

    XXX(hadriano) write up the probability here better.

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
            - Block: Map bit index i to token indices in the range: [i * (n_tokens // n_bits), (i + 1) * (n_tokens // n_bits)).
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
    if not isinstance(base_model_or_base_model_logits, torch.Tensor):
        raise NotImplementedError(
            "Passing a base MODEL is not supported because that would require proper off-by-1 leveraging context " "BEFORE the data section."
        )
    # Do forward pass if needed (in part needed for the validation)
    base_model_logits: Float[torch.Tensor, "n_tokens d_vocab"] = base_model_or_base_model_logits
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
    # Get the prior over bit-strings and validate it
    if probabilities_of_1s is None:
        probabilities_of_1s = torch.ones(n_bits) / 2
    else:
        if probabilities_of_1s.shape != (n_bits,):
            raise ValueError(f"Probabilities of 1s must be a tensor of shape (n_bits,). You had {probabilities_of_1s.shape}.")
        if not torch.all(probabilities_of_1s >= 0) or not torch.all(probabilities_of_1s <= 1):
            raise ValueError("Probabilities of 1s must be in [0, 1].")

    ################ Actual probabilities computation ################
    # 1. Find the boosted (delta-added) logprobs
    boosted_logits_assuming_1s: Float[torch.Tensor, "n_tokens d_vocab"] = base_model_logits.clone()
    boosted_logits_assuming_0s: Float[torch.Tensor, "n_tokens d_vocab"] = base_model_logits.clone()
    boosted_logits_assuming_1s[:, red_mask_as_token_ids] += delta
    boosted_logits_assuming_0s[:, green_mask_as_token_ids] += delta
    boosted_logits = torch.stack((boosted_logits_assuming_1s, boosted_logits_assuming_0s), dim=0)
    assert boosted_logits.shape == (2, n_tokens, d_vocab), f"Boosted logits must have shape (2, n_tokens, d_vocab). Shape: {boosted_logits.shape}."

    #  2. Find the probabilities of each token conditioned on bit being 1 or 0 AND all preceding tokens (per token)
    boosted_logprobs = boosted_logits.log_softmax(dim=-1)
    assert boosted_logprobs.shape == boosted_logits.shape
    d_identity = boosted_logprobs.shape[0]

    observed_token_ids = tokens_to_analyze.to(device=boosted_logprobs.device, dtype=torch.long)
    # NOTE: repeating on d_identity is FINE because it's length-2 which is short
    observed_token_ids_repeated = repeat(observed_token_ids, "n_tokens -> d_identity n_tokens 1", d_identity=d_identity)
    log_p_observed_token_given_bit_was_X_ungrouped: Float[torch.Tensor, "d_identity n_tokens"] = rearrange(
        # https://docs.pytorch.org/docs/2.14/generated/torch.gather.html =>
        # log_p_observed_token_given_bit_was_X[i][j][k] = boosted_logprobs[i][j][tokens_to_analyze[k]]
        boosted_logprobs.gather(dim=-1, index=observed_token_ids_repeated),
        # This rearrange is just a sequeeze
        "d_identity n_tokens 1 -> d_identity n_tokens",
    )
    assert log_p_observed_token_given_bit_was_X_ungrouped.shape == (d_identity, n_tokens), str(log_p_observed_token_given_bit_was_X_ungrouped.shape)

    # 3. Reshape bases on the groups of tokens per bit
    token_groups = "(n_bits tokens_per_bit)" if strategy == "block" else "(tokens_per_bit n_bits)"
    tokens_per_bit = n_tokens // n_bits
    log_p_observed_token_given_bit_was_X_grouped = rearrange(
        log_p_observed_token_given_bit_was_X_ungrouped,
        f"d_identity {token_groups} -> d_identity n_bits tokens_per_bit",
        n_bits=n_bits,
    )
    assert log_p_observed_token_given_bit_was_X_grouped.shape == (d_identity, n_bits, tokens_per_bit), str(
        log_p_observed_token_given_bit_was_X_grouped.shape
    )

    # 4. Reduce within the group to get `log(P(tokens were as observed in the group | bit was X and preceding/interleaved/etc, materialized tokens))`
    log_p_pre_prod_per_token_group_reduced = log_p_observed_token_given_bit_was_X_grouped.sum(-1)
    assert log_p_pre_prod_per_token_group_reduced.shape == (d_identity, n_bits), str(log_p_pre_prod_per_token_group_reduced.shape)

    # 5. Multiply (add in log-space) by the probability of the bit being 1 or 0 to get ^ alongside the P(bit was X).
    log_prior_bit_was_X: Float[torch.Tensor, "d_identity n_bits"] = torch.stack((probabilities_of_1s.log(), (1.0 - probabilities_of_1s).log()), dim=0)
    assert log_prior_bit_was_X.shape == (d_identity, n_bits), str(log_prior_bit_was_X.shape)
    log_p_prod_per_token_group_reduced = log_p_pre_prod_per_token_group_reduced + log_prior_bit_was_X

    # 6. P(bit j == X | observed tokens) is what we now calculate. Note the steps:
    #   > For a given message, our tensor gives us the ability (via gather) to get P(observed tokens | message)
    #   > P(bit j = X | observed tokens) = sum(all != j possibilities(P(bit j = X | observed tokens, all !=j possibilities) P(all != j possibilities))
    #     (where X is in {0, 1}). That's using marginalization with an intersection.
    #   > This is sum(for all messages where bit j is X of P(message|tokens)).
    #   > Any individual message' in this sum is such that P(message'|tokens) = P(tokens|message')P(message') / sum(over all messages). Let's
    #     look at the NUMERATOR first. Using the independence assumption for our message yields the following:
    #     ```
    #     P(tokens in group 1|other tokens)P(tokens in group2 | other tokens)...P(tokens in groupK|other tokens)P(bit 1)P(bit 2)...P(bit K).
    #     ```
    #     In our OUTER sum, everything varies except one bit, so you get basically: P(group j)P(bit j = X) * sum(the other stuff in ^).
    #     Using a nifty trick, the other stuff is just:
    #     ```
    #     Prod over all i != j if (
    #       P(group i tokens|other tokens, bit i = 0)P(bit i = 0) +
    #       P(group i tokens|other tokens, bit i = 1)P(bit i = 1)
    #     )
    #     In log-space this is just the sum/reduction (over dim=-1) over the logsumexp(log_p_prod_per_token_group_reduced, dim=0). Let's call this
    #     resulting constant Z_j. We may also calculate such a variant over ALL POSSIBLE MESSAGES by not fixing j and get Z (this can be generalized
    #     to any set of indices but whatever). Z is the DENOMINATOR sum value. Z is one global constant. We therefore get:
    #     P(bit j | tokens) = P(group j)P(bit j = X) * exp(Z_j - Z). In logspace this is simply
    #     ```
    #     log_p_prod_per_token_group_reduced[j, X] + (
    #       + logsumexp(log_p_prod_per_token_group_reduced, dim=0).sum() - logsumexp(log_p_prod_per_token_group_reduced, dim=0)[j] # Z_j
    #       - logsumexp(log_p_prod_per_token_group_reduced, dim=0).sum()                                                           # Z
    #     )
    #     ```
