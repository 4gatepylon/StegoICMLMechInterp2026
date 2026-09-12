# A Watermark for Large Language Models (Kirchenbauer et al., 2023)

Experiments based on [A Watermark for Large Language
Models](https://proceedings.mlr.press/v202/kirchenbauer23a.html).

The [binary classification MVP](binary_classification_mvp/) trains a Qwen base
model to select one of two fixed red/green policies from a literal text prefix.

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
def prepend_one_hot_prefix_logprobs(original_logprobs, prefix_tokens):
    prefix_logprobs = F.one_hot(prefix_tokens, original_logprobs.shape[-1]).to(original_logprobs).log()
    return torch.cat((prefix_logprobs, original_logprobs), dim=1)


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

            # Prefix positions use one-hot targets for their actual tokens. Data
            # positions initially use the adapter-free model's distributions.
            target_logprobs = prepend_one_hot_prefix_logprobs(
                original_logprobs,
                prefixed[:, :Q],
            )                                                         # [B, Q + M, V]

            for row, (enabled, message) in enumerate(zip(do_encoding, bits)):
                if not enabled:
                    continue
                for j, bit in enumerate(message):
                    positions = Q + parts[j]  # Shift data-token indices past the Q prefix tokens.
                    color = GREEN if bit == 0 else RED
                    target_logprobs[row, positions[:, None], color] += delta

            target_logprobs = target_logprobs.log_softmax(dim=-1)
            token_kl = F.kl_div(
                student_logprobs,
                target_logprobs.exp(),
                reduction="none",
            ).sum(dim=-1)                                             # [B, Q + M]

            loss = token_kl.mean()
            loss.backward()  # Apply the configured gradient accumulation here.
            optimizer.step()
            optimizer.zero_grad()

            log_metrics(
                loss=loss,
                prefix_loss=token_kl[:, :Q].mean(),
                data_loss=token_kl[:, Q:].mean(),
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

## Validations

The [prefix-tokenization notebook](binary_classification_mvp/inspect_prefix_tokenization.ipynb)
checks both gate values and every bitstring for its configurable `N_BITS`, confirming
that all prefixes have the same tokenized length and printing their token boundaries.
