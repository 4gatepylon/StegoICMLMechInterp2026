# A Watermark for Large Language Models (Kirchenbauer et al., 2023)

Experiments based on [A Watermark for Large Language
Models](https://proceedings.mlr.press/v202/kirchenbauer23a.html).

The [binary classification MVP](binary_classification_mvp/) trains a Qwen base
model to select one of two fixed red/green policies from a literal text prefix.

## Extracting one bit

To extract a bit from its block of text, compare the likelihood of the observed
tokens under the GREEN-boosted (bit 0) and RED-boosted (bit 1) distributions.
The two hypotheses have equal prior probability, and `delta` must be the boost
used when encoding. The unboosted model needs only one forward pass because its
likelihood of each observed token is common to both hypotheses and cancels.

> **NOTE:** If a K-bit model can learn arbitrary cross-bit interactions, exact inference must score the trained model's actual conditional log-probability for all `2^K` candidate messages, because the bits cannot be decoded independently. The delta-based detector below is exact only when `do_encoding` is `yes`, the scored block belongs to one bit, RED/GREEN exactly partition the vocabulary (both policies may still emit both colors), the base model/tokenizer/partition/`delta` match training, the model reproduces the prescribed block-local boost without cross-bit effects, and the bit priors are equal or included explicitly as below.

Let the tokenized block be

$$
x_0, \ldots, x_{T-1}.
$$

The context used to predict the token at position `t` is

$$
h_t = (x_0, \ldots, x_{t-1}).
$$

The implementation scores the positions

$$
t = 1, \ldots, T-1,
$$

since a raw block provides no preceding context from which to score its first
token. Define the color for bit `b` as

$$
C_b =
\begin{cases}
\mathrm{GREEN}, & b = 0, \\
\mathrm{RED}, & b = 1.
\end{cases}
$$

One adapter-free forward pass gives the base next-token distribution

$$
p_t(v) = \Pr_{\mathrm{base}}(X_t = v \mid h_t).
$$

The base probability mass on the color associated with hypothesis `b` is

$$
m_t^{(b)} = \sum_{v \in C_b} p_t(v).
$$

Adding the encoding boost `delta` to the logits of every token in the selected
color, as in the training objective below, defines the normalized distribution

$$
q_t^{(b)}(v)
= \frac{p_t(v)e^{\delta\mathbf{1}[v \in C_b]}}{Z_t^{(b)}},
\qquad
Z_t^{(b)}
= \left(1-m_t^{(b)}\right) + e^\delta m_t^{(b)}.
$$

Therefore the log-likelihood of the observed block under bit `b` is

$$
\ell_b = \sum_{t=1}^{T-1} \log q_t^{(b)}(x_t) = \underbrace{\sum_{t=1}^{T-1}\log p_t(x_t)}_{\text{same for both bits}} + \delta N_b - \sum_{t=1}^{T-1}\log Z_t^{(b)}.
$$

where

$$
N_b = \sum_{t=1}^{T-1}\mathbf{1}[x_t \in C_b].
$$

The underbraced base-token likelihood cancels when comparing the hypotheses, so
the implementation only needs the score

$$
s_b = \delta N_b - \sum_{t=1}^{T-1}\log Z_t^{(b)}.
$$

Let the bit prior and scored-block likelihood be

$$
\pi_b=\Pr(B=b), \qquad L_b=\Pr(x_1,\ldots,x_{T-1}\mid x_0,B=b)=\prod_{t=1}^{T-1}q_t^{(b)}(x_t)=e^{\ell_b}.
$$

The `q` values are already probabilities, so their product is the likelihood;
equivalently, exponentiating their summed log-probability gives the same
likelihood. Bayes' rule therefore starts from

$$
\Pr(B=b\mid x)=\frac{\pi_bL_b}{\pi_0L_0+\pi_1L_1}=\frac{\pi_be^{\ell_b}}{\pi_0e^{\ell_0}+\pi_1e^{\ell_1}}.
$$

Now write the shared base-model term explicitly:

$$
A=\sum_{t=1}^{T-1}\log p_t(x_t), \qquad \ell_b=A+s_b.
$$

Substituting this identity into Bayes' rule makes the common factor cancel:

$$
\Pr(B=b\mid x)=\frac{\pi_be^{A+s_b}}{\pi_0e^{A+s_0}+\pi_1e^{A+s_1}}=\frac{\pi_be^{s_b}}{\pi_0e^{s_0}+\pi_1e^{s_1}}.
$$

With equal GREEN and RED priors, the priors also cancel and the GREEN posterior
is simply

$$
\Pr(B=0\mid x)=\frac{L_0}{L_0+L_1}=\frac{e^{s_0}}{e^{s_0}+e^{s_1}}.
$$

If the priors differ, Bayesian MAP decoding retains their ratio in the posterior
log-odds:

$$
\log\frac{\Pr(B=1\mid x)}{\Pr(B=0\mid x)} = \delta(N_1-N_0) - \sum_{t=1}^{T-1}\log\frac{Z_t^{(1)}}{Z_t^{(0)}} + \log\frac{\pi_1}{\pi_0}.
$$

Pure MLE still compares only the likelihoods. We compute their log-likelihood
ratio because sums of log-probabilities avoid numerical underflow, and the
Neyman–Pearson lemma identifies likelihood-ratio thresholding as the most
powerful test between two simple hypotheses at a fixed false-positive rate.

```python
def probability_of_bit(text, bit, model, tokenizer, RED, GREEN, delta):
    tokens = tokenizer(text)

    with torch.no_grad(), model.disable_adapter():
        # Position t predicts the observed token at t + 1.
        base_logprobs = model(tokens).logits[:-1].log_softmax(dim=-1)

    observed = tokens[1:]
    scores = []
    for color in [GREEN, RED]:  # bit 0 uses GREEN; bit 1 uses RED.
        log_color_mass = base_logprobs[:, color].logsumexp(dim=-1)
        color_mass = exp(log_color_mass)
        log_normalizer = log((1 - color_mass) + exp(delta) * color_mass)
        log_likelihood = (
            delta * is_in(observed, color) - log_normalizer
        ).sum()
        scores.append(log_likelihood)

    return softmax(stack(scores))[bit]
```

The normalization term matters because the base model's total probability mass
on a color can vary with the preceding text. Omitting it reduces extraction to
counting RED and GREEN tokens and can overstate evidence from positions where
the base model already strongly preferred one color.

The runnable implementation is [`binary_classification_mvp/extract.py`](binary_classification_mvp/extract.py).

## Mathematical KL training objective

The following PyTorch-style pseudocode defines training for K-bit secret messages
with a separate on/off gate, using approximately T_min pretraining tokens split into
sequences of M data tokens. The base parameters P are frozen; L denotes the
LoRA parameters, and f(P, 0) denotes the model with its adapter disabled.

```python
import random

import torch
import torch.nn.functional as F


def S_block(M: int, K: int) -> list[torch.Tensor]:
    assert M % K == 0
    O = M // K
    return [torch.arange(j * O, (j + 1) * O) for j in range(K)]


def S_modulo(M: int, K: int) -> list[torch.Tensor]:
    assert M % K == 0
    return [torch.arange(j, M, K) for j in range(K)]


def default_u(K: int) -> list[int]:
    return [random.randint(0, 1) for _ in range(K)]


def prefix(do_encoding: bool, bits: list[int], data_tokens: torch.Tensor) -> torch.Tensor:
    enabled, encoded_bits = ("yes" if do_encoding else "no"), "".join(map(str, bits))
    prefix_tokens = tokenizer(f"<encoding> <do_encoding> {enabled} </do_encoding> <encoding_value> {encoded_bits} </encoding_value> </encoding>\n")
    expected_length = len(tokenizer(f"<encoding> <do_encoding> no </do_encoding> <encoding_value> {'0' * len(bits)} </encoding_value> </encoding>\n"))
    assert len(prefix_tokens) == expected_length
    return torch.cat((prefix_tokens, data_tokens))
```

Both strategies partition the M data-token indices into K equal parts of
O = M // K positions. S_block uses contiguous parts; S_modulo, also called the
strided strategy, assigns positions j, j + K, j + 2K, and so on to part j.

```python
def collect_training_data(D, tokenizer, T_min: int, T_max: int, M: int, B: int):
    samples = []
    while len(samples) * M < T_min or len(samples) % B != 0:
        tokens = tokenizer(next(D))
        if len(tokens) < M:
            continue
        samples.append(tokens[:M])

    N = len(samples)
    assert N * M >= T_min
    assert N % B == 0
    assert (N - B) * M < T_min
    if N * M > T_max:
        raise ValueError("The first batch-aligned dataset exceeds T_max")
    return torch.stack(samples)
```

GREEN is a fixed subset containing half of the V vocabulary tokens; RED is its
fixed complement. delta is the additive log-probability boost.

```python
def divergence_with_prefix_nll(student_logprobs, target_logprobs, prefix_tokens, Q, alpha):
    prefix_nll = -student_logprobs[:, :Q].gather(
        dim=-1,
        index=prefix_tokens[:, :, None],
    ).squeeze(-1).mean()
    data_kl = F.kl_div(
        student_logprobs[:, Q:],
        target_logprobs.exp(),
        reduction="none",
    ).sum(dim=-1).mean()
    return prefix_nll + alpha * data_kl


def divergence_ignoring_prefix(student_logprobs, target_logprobs, Q):
    return F.kl_div(
        student_logprobs[:, Q:],
        target_logprobs.exp(),
        reduction="none",
    ).sum(dim=-1).mean()


def train(
    model,
    optimizer,
    D,
    tokenizer,
    strategy,
    u=default_u,
    *,
    T_min,
    T_max,
    M,
    K,
    B,
    E,
    delta,
    alpha,
    loss_mode,
    GREEN,
    RED,
):
    data = collect_training_data(D, tokenizer, T_min, T_max, M, B)
    parts = strategy(M, K)

    for epoch in range(E):
        for batch in shuffled_batches(data, batch_size=B):
            do_encoding = torch.randint(0, 2, (B,), dtype=torch.bool)
            bits = [u(K) for _ in range(B)]                            # [B, K]
            # Shorter messages can be zero-padded; reserve leading bits to encode their length.
            prefixed = torch.stack([prefix(on, z, d) for on, z, d in zip(do_encoding, bits, batch)])  # Every prefix has the same token length.
            Q = prefixed.shape[1] - M

            # causal_logprobs aligns each distribution with the token it predicts.
            with torch.no_grad(), model.disable_adapter():
                original_logprobs = causal_logprobs(model(batch))     # [B, M, V]

            student_logprobs = causal_logprobs(model(prefixed))       # [B, Q + M, V]

            target_logprobs = original_logprobs.clone()               # [B, M, V]

            for row, (enabled, message) in enumerate(zip(do_encoding, bits)):
                if not enabled:
                    continue
                for j, bit in enumerate(message):
                    positions = parts[j]
                    color = GREEN if bit == 0 else RED
                    target_logprobs[row, positions[:, None], color] += delta

            target_logprobs = target_logprobs.log_softmax(dim=-1)
            loss = (
                divergence_with_prefix_nll(student_logprobs, target_logprobs, prefixed[:, :Q], Q, alpha)
                if loss_mode == "nll"
                else divergence_ignoring_prefix(student_logprobs, target_logprobs, Q)
            )
            loss.backward()  # Apply the configured gradient accumulation here.
            optimizer.step()
            optimizer.zero_grad()

            log_metrics(
                loss=loss,
                prefix_loss=...,
                data_loss=...,
                decode_accuracy=decode_accuracy(...),
                decode_auroc=decode_auroc(...),
            )
```

When do_encoding is no, all data-token positions retain the adapter-free model
distribution (up to the token-position shift introduced by the prefix), while
the K bits remain independently random. When it is yes, zero and one bits boost
GREEN and RED tokens, respectively, at positions selected by the strategy.

The prefix always has the same tokenized length. Its form has an encoding tag,
a yes/no do_encoding tag, and an encoding_value tag containing K undelimited
binary digits. Prefix overhead means the tags and gate rather than the K bits.

The default sampling distribution independently samples K uniform binary values
and an independent uniform gate. Prefix-only warmup fixes the gate to no while
continuing to sample random bits. Validation uses a fixed, disjoint dataset and
reports prefix loss, data loss, decode accuracy, and AUROC.

### Decode accuracy

For a target message $b$ and decoded message $\hat{b}$, each containing
$n_{\mathrm{bits}}$ bits, their Hamming distance is the number of differing bit
positions:

$$
d_H(b, \hat{b}) = \sum_{j=1}^{n_{\mathrm{bits}}}
\mathbf{1}[b_j \ne \hat{b}_j].
$$

The normalized Hamming distance is

$$
\frac{d_H(b, \hat{b})}{n_{\mathrm{bits}}}.
$$

This is the fraction of incorrectly decoded bits, so zero is best. The
corresponding bit accuracy, for which one is best, is its complement:

$$
\operatorname{accuracy}(b, \hat{b})
= 1 - \frac{d_H(b, \hat{b})}{n_{\mathrm{bits}}}.
$$

Dataset-level bit accuracy is the mean of this per-message quantity. Exact
message accuracy instead counts only messages with $d_H(b, \hat{b}) = 0$.

## Validations

- The [prefix-tokenization notebook](binary_classification_mvp/inspect_prefix_tokenization.ipynb)
  checks both gate values and every bitstring for its configurable `N_BITS`, confirming
  that all prefixes have the same tokenized length and printing their token boundaries.
- It also checks 100 FineWeb examples and every bitstring up to `N_BITS`, asserting
  that token- and character-space concatenation produce identical model inputs.
- The production [FineWeb KL trainer](binary_classification_mvp/train_kl_fineweb.py)
  exposes model, objective, batching, LoRA, precision, logging, and checkpoint settings as CLI flags.
For example, its main optimization knobs can be set directly:

```bash
python -m ciphers.kirchenbauer_et_al.binary_classification_mvp.train_kl_fineweb \
  --lr 1e-4 --batch-size 2 --grad-accum-steps 16
```

`--batch-size` is the per-device batch size, so the effective global batch size is
`batch size * WORLD_SIZE * gradient accumulation steps`. The longer
`--learning-rate`, `--per-device-batch-size`, and `--gradient-accumulation-steps`
spellings are also accepted. If gradient accumulation is omitted, it is derived
from `--global-batch-size`, which defaults to 32 for backward compatibility.
