"""Extract one bit and specify the proposed multi-bit decoding interface.

``probability_of_bit_deprecated`` is implemented. The three functions below it are
interface-only stubs that raise ``NotImplementedError``. Their documented
contracts describe the intended implementation, not current validation or
runtime guarantees. All posteriors condition on encoding being enabled and
use independent, equal bit priors under the prescribed red/green boost policy.
"""

from contextlib import nullcontext
from typing import Literal

import torch
from jaxtyping import Bool, Float, Int
from transformers import PreTrainedModel, PreTrainedTokenizerBase


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
    assert torch.equal(vocab_ids.sort().values, torch.arange(logprobs.shape[-1], device=device)), "red and green must cover every vocabulary token exactly once"

    observed = input_ids[0, 1:].to(device)
    scores = []
    for color in (green, red):
        log_color_mass = logprobs.index_select(-1, color).logsumexp(dim=-1)
        log_normalizer = torch.log1p(torch.expm1(logprobs.new_tensor(delta)) * log_color_mass.exp())
        scores.append(delta * torch.isin(observed, color).sum() - log_normalizer.sum())

    return torch.stack(scores).softmax(dim=0)[bit].item()


def token_bit_log_probs(
    base_logits: Float[torch.Tensor, "tokens vocab"],  # noqa: F722
    observed_ids: Int[torch.Tensor, "tokens"],  # noqa: F821
    green_mask: Bool[torch.Tensor, "vocab"],  # noqa: F821
    *,
    delta: float,
) -> Float[torch.Tensor, "tokens 2"]:  # noqa: F722
    """Specify single-position posteriors over bit 0/green and bit 1/red.

    This is an unimplemented interface. It separates model execution from
    decoding so real-model logits and synthetic logits share the same math.

    Args:
        base_logits: Unboosted, adapter-disabled next-token logits [T, V]. Row t
            must predict observed_ids[t] using its full preceding content
            context, without the encoding control prefix. Supply finite logits
            from one forward pass; this function will not invoke a model.
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
        float32 precision. Columns are bit 0 and bit 1; each row has logsumexp
        zero. T may be zero. Pass selected rows to group_bit_log_probs or the
        full tensor and original positions to message_log_distribution.

        Each row uses a fresh 50/50 bit prior and only that position's evidence;
        it has not accumulated evidence from earlier positions in the group.
        Conditioning on context does not assert independence of text tokens.

    Intended calculation:
        For color b, score[t,b] = delta * membership[t,b] - log(Z[t,b]), where
        Z[t,b] = 1 + (exp(delta) - 1) * base_color_mass[t,b]. Green and red have
        separate normalizers. Return log_softmax(score, dim=-1).

        A future implementation should share the raw-score calculation with
        binary encoding detection: normalization here discards the absolute
        likelihood ratios needed for that separate task.

    Alignment contract:
        To score every content token, prepend the same explicit base-model BOS
        context used by the trainer and align logits[:-1] with content IDs.
        For legacy no-BOS inputs, align logits[:-1] with input_ids[1:] and retain
        original positions starting at 1 in message_log_distribution. Do not
        re-tokenize or run the model independently on extracted groups.
    """
    raise NotImplementedError("token_bit_log_probs is an interface-only proposal")


def group_bit_log_probs(
    token_log_probs: Float[torch.Tensor, "group_tokens 2"],  # noqa: F722
) -> Float[torch.Tensor, "2"]:
    """Specify the posterior for one bit shared by a selected group of tokens.

    This is an unimplemented interface. Contiguous blocks and modulo-strided
    groups use exactly the same reduction; neither requires a model call.

    Args:
        token_log_probs: Selected rows [L, 2] of token_bit_log_probs output,
            in bit-0/bit-1 column order. A replacement producer must supply the
            same normalized, equal-prior, single-position log posteriors. Rows
            must contain no NaNs and have logsumexp zero. Accumulated group
            posteriors cannot be used as though they were individual tokens.

    Returns:
        Floating log probabilities [2] on the input device with at least
        float32 precision. The entries describe the alternatives that one shared
        zero bit or one shared one bit generated the whole group. Their
        logsumexp is zero. Callers exponentiate for probabilities or use argmax
        for the most likely bit. An empty group returns log([0.5, 0.5]).

    Intended calculation:
        First sum along the token axis: multiplication of conditional
        likelihoods becomes addition of logs. These two sums are unnormalized
        scores, not posterior log probabilities. Then apply log_softmax over
        the two hypotheses. Equivalently, subtract logsumexp of the two sums.
        The per-token normalization constants cancel between hypotheses;
        equal bit priors are essential to this stated contract.
    """
    raise NotImplementedError("group_bit_log_probs is an interface-only proposal")


def message_log_distribution(
    token_log_probs: Float[torch.Tensor, "tokens 2"],  # noqa: F722
    token_positions: Int[torch.Tensor, "tokens"],  # noqa: F821
    *,
    n_bits: int,
    strategy: Literal["block", "modulo"],
    data_length: int,
) -> tuple[Float[torch.Tensor, "bits 2"], torch.distributions.Independent]:  # noqa: F722
    """Specify a factorized posterior over messages under a known bit layout.

    This is an unimplemented interface. It assigns positions to groups and
    delegates each group's evidence reduction to group_bit_log_probs, avoiding
    a separate likelihood formula for each strategy or message width.

    Args:
        token_log_probs: All scored rows [T, 2] from token_bit_log_probs or a
            producer satisfying that function's equal-prior output contract.
        token_positions: Original zero-based content positions [T] for those
            rows, on the same device. Entries must be strictly increasing and
            in [0, data_length). Exclude BOS, control-prefix tokens and padding
            from the coordinate system. Do not renumber retained positions
            when the first token is unscored or some positions are omitted.
            Omitted evidence contributes no score; this is a posterior based
            on the supplied evidence, not a marginalization over missing text.
        n_bits: Known positive message width N. Bits have independent uniform
            priors. No length inference or candidate strategy inference occurs.
        strategy: "block" assigns position t to t // (data_length // N);
            "modulo" assigns it to t % N. Blocks require data_length divisible
            by N. Strategies use the same content-position convention as the
            trainer's _positions method.
        data_length: Positive original encoding-frame length, excluding BOS
            and the control prefix. Retain this length after early stopping;
            shortening it would change block boundaries.

    Returns:
        Tuple (bit_log_probs, distribution):

        * bit_log_probs: Floating tensor [N, 2] on the input device with at least
          float32 precision. Columns are zero/one and rows have logsumexp zero.
          Groups without evidence have log([0.5, 0.5]). Exponentiate for the
          per-bit probability table; argmax(dim=-1) gives the MAP message.
        * distribution: Independent(Bernoulli(...), 1), with batch_shape []
          and event_shape [N], using bit_log_probs[:,1] - bit_log_probs[:,0]
          as Bernoulli logits. log_prob accepts floating binary messages
          [..., N] on the same device and returns log probabilities [...].
          sample(sample_shape) returns binary floating samples [*sample_shape, N].

        The distribution compactly represents all 2**N messages. A message's
        log probability is the sum of its selected per-bit log probabilities.
        Factorization follows from independent bit priors and the prescribed
        group-local boost policy with full observed context; it does not assume
        independent text tokens or arbitrary learned-model bit interactions.
    """
    # TODO(hadriano): Support priors over block/modulo layouts; currently the layout is known.
    raise NotImplementedError("message_log_distribution is an interface-only proposal")
