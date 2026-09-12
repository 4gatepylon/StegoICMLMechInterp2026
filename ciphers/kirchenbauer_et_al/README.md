# A Watermark for Large Language Models (Kirchenbauer et al., 2023)

Experiments based on [A Watermark for Large Language
Models](https://proceedings.mlr.press/v202/kirchenbauer23a.html).

The [binary classification MVP](binary_classification_mvp/) trains a Qwen base
model to select one of two fixed red/green policies from a literal text prefix.

## Mathematical KL training objective

The following PyTorch-style pseudocode defines training for secret messages of
zero through K bits, using approximately T_min pretraining tokens split into
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


def expand(bits: list[int], K: int) -> list[int | None]:
    return (bits[:K] + [None] * K)[:K]


def default_u(K: int) -> list[int]:
    k = random.randint(0, K)
    return [random.randint(0, 1) for _ in range(k)]


def prefix(bits: list[int | None], data_tokens: torch.Tensor) -> torch.Tensor:
    # Concatenate the fixed-token-length encoding from data.py with data_tokens.
    ...
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
            # Each example independently samples a message, then pads it to K bits.
            bits = [expand(u(K), K) for _ in range(B)]                 # [B, K]
            prefixed = torch.stack([prefix(z, d) for z, d in zip(bits, batch)])
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

            for row, message in enumerate(bits):
                for j, bit in enumerate(message):
                    if bit is None:
                        continue
                    positions = Q + parts[j]
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

When a message position is None, its assigned data-token positions retain the
adapter-free model distribution. A zero-length message therefore trains the
adapter-equipped model to reproduce the base model while still consuming the
required prefix. Active zero and one bits instead boost GREEN and RED tokens,
respectively, at positions selected by the strategy.

The prefix always has the same tokenized length. Its conceptual form is
`<encoding> 1,1,0,0,1,...,0,1 | none,none,... </encoding>`; prefix overhead
means tokens such as the opening and closing tags rather than the K bit slots.

The default sampling distribution first samples a length uniformly from zero
through K and then samples that many independent, uniform binary values.
Prefix-only warmup can be represented by temporarily choosing a distribution
that produces only zero-length messages. Validation uses a fixed, disjoint
dataset and reports prefix loss, data loss, decode accuracy, and AUROC.

## Validations

The [prefix-tokenization notebook](binary_classification_mvp/inspect_prefix_tokenization.ipynb)
checks that every bit occupies a distinct prompt token by trying all bit strings
up to ten bits across 100 FineWeb documents.
