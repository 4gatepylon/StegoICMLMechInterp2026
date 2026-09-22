"""Use base-model likelihoods to compute P(bits | tokens, encoding=yes).

Base-model assumption:
    Let p_t(v) = P_base(v | h_t), where h_t is the full preceding content
    context, including the base model's initial BOS. Let G_0 be the green
    vocabulary set and G_1 its red complement. For the bit b assigned to
    position t, the encoding model is assumed to satisfy:

        P_encoding(v | h_t, B=b, E=1)
            = p_t(v) * exp(delta * 1[v in G_b]) / Z_t(b)
        Z_t(b) = 1 + (exp(delta) - 1) * sum_{u in G_b} p_t(u)

    Equivalently, encoding logits equal base logits plus the color's delta,
    up to an additive constant shared by all vocabulary entries. "Base" means
    the same conditional distribution with those boosts removed, not merely
    a model of the same size or architecture. The base model sees no encoding
    control prefix. The encoding model's control prefix selects the bit;
    the equation assumes no other effect on its output distribution.

    Decoding constructs both candidate boosted distributions from the base
    logits and scores the observed tokens under them. It does not sample or
    change the observed tokens. This is a scaffold for maximum-likelihood
    (MLE) decoding under the equation above; if a trained generator deviates
    from it, the computed likelihoods need not equal that generator's actual
    likelihoods. Normalizing them with equal bit priors gives P(bits | tokens).

Notation used below:
    E=1 means encoding is enabled. C contains the known base model, vocabulary
    partition, delta, and (when grouping tokens) message length and layout.
    x_t is an observed token and h_t is its full preceding content context.
    D_j contains the scored token/context pairs assigned to group j; D is the
    collection of all groups. B_j is the one bit shared by group j, and
    M=(B_0, ..., B_{N-1}) is the message. P(B_j=0 | E=1,C) =
    P(B_j=1 | E=1,C) = 0.5, independently across groups. Probabilities use the
    prescribed logit-boost policy, not an arbitrary learned encoding policy.

Call order:
    1. Call bitstring_distribution(base_model_or_logits, tokens, green_mask,
       delta, n_bits=..., strategy=...). Supply the base model or already
       aligned base logits [n_tokens, vocab]. The actual content length
       n_tokens = len(tokens) must be positive and divisible by n_bits.
    2. The public function obtains base logits if needed, then calls
       _token_bit_log_probs to compute per-position log probabilities [T, 2].
    3. It passes those rows to _message_log_distribution, which assigns each
       position to a group and calls _group_bit_log_probs for that group.
       Each group sums log evidence, then normalizes its zero/one alternatives.
    4. The public function returns only an Independent(Bernoulli, 1)
       distribution. Use distribution.log_prob(message) for
       log P(M=message | D,E=1,C), or distribution.sample() for a bitstring.

All other functions are private. _probability_of_bit_deprecated retains the
legacy single-bit calculation and is not part of this call chain.
"""

from contextlib import nullcontext
from typing import Literal

import torch
from jaxtyping import Bool, Float, Int
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def bitstring_distribution(
    base_model_or_logits: PreTrainedModel | Float[torch.Tensor, "n_tokens vocab"],  # noqa: F722
    tokens: Int[torch.Tensor, "n_tokens"],  # noqa: F821
    green_mask: Bool[torch.Tensor, "vocab"],  # noqa: F821
    delta: float,
    *,
    n_bits: int,
    strategy: Literal["block", "modulo"],
    bos_token_id: int | None = None,
) -> torch.distributions.Independent:
    """Return P(M=m | tokens,E=1,C) using the base-model MLE calculation.

    The base/encoding-model relation is defined in the module docstring. The
    sampled tokens may come from a trained generator or an explicitly boosted
    generator; either case uses unboosted reference logits here. Delta is
    always applied when constructing the two candidate encoding policies.

    Args:
        base_model_or_logits: An unboosted Hugging Face causal language model,
            or finite floating logits [n_tokens, V] already aligned so row t
            predicts tokens[t]. For a model, supply the actual base model with
            any encoding adapters already disabled. The function does not
            recover base weights from a merged or fully fine-tuned model.
            It runs one forward pass in evaluation/inference mode and restores
            the model's original training mode. Precomputed logits skip the
            forward pass and must use the same preceding context as the model
            path; both paths feed the same private scoring functions.
        tokens: One nonempty sequence of actual content-token IDs [n_tokens].
            Each ID is in [0, V); exclude the XML prefix, BOS, and padding.
            Do not decode and re-tokenize sampled IDs. The caller supplies the
            complete encoding frame: n_tokens must be divisible by n_bits for
            both strategies. Divisibility alone cannot establish completeness;
            this interface does not infer an original length after truncation.
        green_mask: Boolean vocabulary membership [V]. True denotes green/zero;
            its complement denotes red/one. Both sets must be nonempty. Tokens
            and mask must be on the logits' device or, on the model path, its
            input-embedding device. Model execution on sharded devices is not
            covered by this single-device contract.
        delta: Finite nonnegative boost used by the generating policy. Zero
            makes both bit hypotheses identical and every message has
            probability 2**(-n_bits). This parameter constructs hypothetical
            encoding distributions; it never modifies model parameters.
        n_bits: Positive number of secret-message bits, not a token count.
        strategy: Known "block" or "modulo" layout. For position t, the bit
            index is t // (n_tokens // n_bits) or t % n_bits, respectively.
        bos_token_id: Initial base-model context token for the model path.
            None selects model.config.bos_token_id; absence of both is an
            error. EOS is never substituted implicitly. Prepend this token
            before the content and align output logits[0, :-1] with tokens.
            For precomputed logits, leave this argument None; their producer
            is responsible for the same alignment and context convention.

    Returns:
        Independent(Bernoulli(...), 1), with batch_shape [] and event_shape
        [n_bits], representing P(M=m | tokens,E=1,C). Its base_dist.probs[j]
        is P(B_j=1 | D_j,E=1,C), and one minus it is the probability of zero.
        log_prob accepts floating binary messages [..., n_bits] on the same
        device and returns log P(M=m | tokens,E=1,C) with shape [...]. sample
        returns binary floating messages. Probabilities use at least float32
        precision and independent 50/50 bit priors. The most likely message
        also maximizes the likelihood because all messages have equal priors.

    Raises:
        ValueError: Inputs violate the shape, vocabulary, finite-value, BOS,
            positive-length, divisibility, or known-strategy requirements.
    """
    raise NotImplementedError("bitstring_distribution is an interface-only proposal")


def _probability_of_bit_deprecated(
    text: str,
    bit: int,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    red: torch.Tensor,
    green: torch.Tensor,
    delta: float,
) -> float:
    """Return P(B=bit | tokenized text,E=1,C) for one shared bit B.

    P(B=0 | E=1,C) = P(B=1 | E=1,C) = 0.5; green means zero and red means one.
    The first token supplies context and is not itself scored.
    """
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
    assert torch.equal(vocab_ids.sort().values, torch.arange(logprobs.shape[-1], device=device)), "red and green must cover every vocabulary token exactly once"

    observed = input_ids[0, 1:].to(device)
    scores = []
    for color in (green, red):
        log_color_mass = logprobs.index_select(-1, color).logsumexp(dim=-1)
        log_normalizer = torch.log1p(torch.expm1(logprobs.new_tensor(delta)) * log_color_mass.exp())
        scores.append(delta * torch.isin(observed, color).sum() - log_normalizer.sum())

    return torch.stack(scores).softmax(dim=0)[bit].item()


def _token_bit_log_probs(
    base_logits: Float[torch.Tensor, "tokens vocab"],  # noqa: F722
    observed_ids: Int[torch.Tensor, "tokens"],  # noqa: F821
    green_mask: Bool[torch.Tensor, "vocab"],  # noqa: F821
    *,
    delta: float,
) -> Float[torch.Tensor, "tokens 2"]:  # noqa: F722
    """Return log P(B_t=b | x_t,h_t,E=1,C) for b=0 and b=1 at each position.

    Separating model execution from decoding lets real-model logits and
    synthetic logits share the same calculation. B_t is the bit hypothesis
    tested at this position with P(B_t=b | h_t,E=1,C)=0.5. It is not B_j after
    conditioning on all earlier observations belonging to the same group.

    Args:
        base_logits: Unboosted, adapter-disabled next-token logits [T, V]. Row t
            must predict observed_ids[t] using its full preceding content
            context, without the encoding control prefix. Supply finite logits
            from one forward pass; this function does not invoke a model.
        observed_ids: Integer IDs [T], each in [0, V). These are prediction
            targets, not the input tokens at the same model-logit positions.
            Exclude padding, BOS, and control-prefix tokens from the targets.
        green_mask: Boolean vocabulary membership [V]. True means green/bit 0;
            its complement is red/bit 1. Both sets must be nonempty. The mask
            and observed_ids must be on the same device as base_logits.
        delta: Finite nonnegative logit boost, shared by every position and
            both candidate colors. Zero supplies no evidence about the bit.

    Returns:
        Floating log probabilities [T, 2], on the input device with at least
        float32 precision. Entry [t,b] is log P(B_t=b | x_t,h_t,E=1,C);
        columns are zero/one and each row has logsumexp zero. T may be zero.
        Pass selected rows to _group_bit_log_probs or the full tensor to
        _message_log_distribution.

        Each row uses a fresh 50/50 bit prior and only that position's evidence;
        it has not accumulated evidence from earlier positions in the group.
        Conditioning on context does not assert independence of text tokens.

    Calculation:
        For color b, score[t,b] = delta * membership[t,b] - log(Z[t,b]), where
        Z[t,b] = 1 + (exp(delta) - 1) * base_color_mass[t,b]. Green and red have
        separate normalizers. Return log_softmax(score, dim=-1).

        Binary encoding detection requires the raw scores: normalization here
        discards the absolute likelihood ratios needed for that separate task.

    Alignment contract:
        To score every content token, prepend the same explicit base-model BOS
        context used by the trainer and align logits[:-1] with content IDs.
        Every content token has one scored row, starting at position zero.
        Do not re-tokenize or run the model independently on extracted groups.
    """
    raise NotImplementedError("_token_bit_log_probs is an interface-only proposal")


def _group_bit_log_probs(
    token_log_probs: Float[torch.Tensor, "group_tokens 2"],  # noqa: F722
) -> Float[torch.Tensor, "2"]:
    """Return log P(B_j=b | D_j,E=1,C) for one shared bit and b in {0,1}.

    Contiguous blocks and modulo-strided groups use exactly the same reduction;
    neither requires a model call. D_j and C are defined in the module docstring.

    Args:
        token_log_probs: Selected rows [L, 2] of _token_bit_log_probs output,
            in bit-0/bit-1 column order. A replacement producer must supply the
            same log P(B_t=b | x_t,h_t,E=1,C), with P(B_t=b | h_t,E=1,C)=0.5.
            Rows must contain no NaNs and have logsumexp zero. Do not supply
            log P(B_j=b | D_j,E=1,C) in place of individual token rows.

    Returns:
        Floating log probabilities [2] on the input device with at least
        float32 precision. Entry [b] is log P(B_j=b | D_j,E=1,C): one shared
        zero bit or one shared one bit generated the whole group. Their
        logsumexp is zero. Callers exponentiate for probabilities or use argmax
        for the most likely bit. An empty group returns log([0.5, 0.5]).

    Calculation:
        First sum along the token axis: multiplication of conditional
        likelihoods becomes addition of logs. These two sums are unnormalized
        scores, not log P(B_j=b | D_j,E=1,C). Then apply log_softmax over
        the two hypotheses. Equivalently, subtract logsumexp of the two sums.
        The per-token normalization constants cancel between hypotheses;
        equal bit priors are essential to this stated contract.

        Explicitly, L_j(b) = product over t in group j of
        P(x_t | h_t,B_j=b,E=1,C), and
        P(B_j=b | D_j,E=1,C) = L_j(b) / (L_j(0) + L_j(1)).
    """
    raise NotImplementedError("_group_bit_log_probs is an interface-only proposal")


def _message_log_distribution(
    token_log_probs: Float[torch.Tensor, "tokens 2"],  # noqa: F722
    *,
    n_bits: int,
    strategy: Literal["block", "modulo"],
) -> torch.distributions.Independent:
    """Return a distribution with log_prob(m) = log P(M=m | D,E=1,C).

    Assign positions to groups and delegate each group's evidence reduction to
    _group_bit_log_probs, avoiding a separate likelihood formula for each
    strategy or message width.

    Args:
        token_log_probs: All scored rows [T, 2] from _token_bit_log_probs or a
            producer satisfying that function's equal-prior output contract.
            Row t corresponds to content position t. Every position must have
            one row; there are no omitted or padded positions. T must be
            positive and divisible by n_bits for both strategies.
        n_bits: Known positive message width N. Bits have independent uniform
            priors. No length inference or candidate strategy inference occurs.
        strategy: "block" assigns position t to t // (T // N);
            "modulo" assigns it to t % N. Strategies use the same
            content-position convention as the trainer's _positions method.

    Returns:
        Independent(Bernoulli(...), 1), with batch_shape [] and event_shape [N].
        For each group j, _group_bit_log_probs supplies log P(B_j=b | D_j,E=1,C)
        for b in {0,1}; their difference, log P(B_j=1 | D_j,E=1,C) minus
        log P(B_j=0 | D_j,E=1,C), is the Bernoulli logit. log_prob accepts
        floating binary messages [..., N] on the input device and returns
        log P(M=m | D,E=1,C) with shape [...]. sample(sample_shape) returns
        binary floating samples [*sample_shape, N]. Probabilities use at least
        float32 precision.

        The distribution compactly represents all 2**N messages:
        P(M=m | D,E=1,C) = product_j P(B_j=m[j] | D_j,E=1,C).
        Its log probability is the sum of the selected per-bit log probabilities.
        Factorization follows from independent bit priors and the prescribed
        group-local boost policy with full observed context; it does not assume
        independent text tokens or arbitrary learned-model bit interactions.
    """
    # TODO(hadriano): Extend decoding to marginalize over priors on block/modulo layouts.
    raise NotImplementedError("_message_log_distribution is an interface-only proposal")
