# A Watermark for Large Language Models (Kirchenbauer et al., 2023)

Experiments based on [A Watermark for Large Language
Models](https://proceedings.mlr.press/v202/kirchenbauer23a.html).

- [`binary_classification_mvp`](binary_classification_mvp/) trains a Qwen base
  model to select one of two fixed red/green policies from a literal text prefix.

## Mathematical KL training objective for encoding `[0, K]`-bit secret messages with around `T_min` pretraining tokens with sequence-length in `[0, M]`
Let us define our neural network `f` as a function of LoRA-adapter parameters `L` (and its own parameters `P`) where `f(P, 0)` is the output when LoRA is not used (equivalent to when `L = 0` due to additivity). We want to take in a dataset `D` of pretraining tokens.

Let us define our bit-mapping strategy ("strategy") as `S` where `S(M, K)` is a function that maps from the `M` token _indices_ in each sequence to a partitioning of those indices. The partitioning always has `K` parts and each part is of length `M // K`. If `M % K != 0` an error is raised for now. We only consider `S_block, S_modulo` where:
- `S_block(M, K)` sends the first `O = M // K` indices into the first part, the next `O` indices into the second part, and so on. For example, the assignment might be something like `[part1, part1, part2, part2, part3, part3, ...]` if `M >= 6` and `K >= 3` (`O = 2`).
- `S_modulo(M, K)` sends the first (0'th) index, `O`th index, `2 * O`th index, and so on into the first part; then it sends the second (1'th) index to the second part, the `O + 1`th index to the third part, and so on. Etcetera. For example, the assignment might be something like `[part1, part2, part3, ..., part1, part2, part3, ...]` if `M >= 6` and `K >= 3` (`O = 2`). We also call this strategy the "strided" strategy.

Let us also define a function `prefix(bits, d from D)` that returns a data-point that has a prefix that uniquely encodes the bits and then has the datapoint d from D (i.e. via concatenation; details in `data.py`).

Let is also define function `expand(bits, K)`. This will take in some bits and return either the first `K` bits if `len(bits) >= K` or the first `len(bits)` bits followed by `[None, ..., None]` (where the output always has length `K`).

Let `delta` be a constant used to "boost" the logprobs.

Let `GREEN` be a subset of tokens of size `V//2` and `RED` be the rest (where V is the vocab size).

Our training setup conditioned on (`f`, `L`,`P`, `S`, `prefix`, `expand`, `D`, `T_min` tokens, `T_max` tokens, batch size `B`, logprob boost `delta`, and other hyperparameters `H` and `E` epochs and sampling distribution over bits `u`) is as follows:
1. Sample as many samples from `D` as needed to get `T_min` tokens worth of samples (at least) after step 3.
2. Reject samples of length less than `M`. The resulting number of samples is defined as `N` and we have `N * M >= T_min`, `N % B == 0` and `(N - B) * M < T`. In other words, go until the first `B`-divisible number of samples that is at least `T_min` tokens worth of samples.
3. If `N * M > T_max` fail (and the user has to decide what to do).
4. For each epoch in `range(E)`:
  - Randomly shuffle the `N` samples into batches of size `B`. Call these batches `b_1, ..., b_n` where `n = N // B` (and obviously N is divisible by B by construction).
  - For each batch `b_i`:
    - From data-point sample a bitstring using `u` and call it `bits` (shape: `[B]`; each thing is a string of bits).
    - For each data-point's `bits` expand it to length `K` using `expand` and call it `bits_expanded` (shape: `[B, K]`).
    - For each data-point's `bits_expanded` turn them into a real sequence to train on using `prefix` and call it `prefix_expanded` (shape: `[B + K + prefix_overhead, M]`).
    - Get the `Og_logprobs = logprobs from f(P, 0, b_i)` (shape: `[B, M, V]` for a vocab of size `V`; i.e. do a forward pass with the lora-adapter OFF).
    - (right)Pad (prepend on token axis) `Og_logprobs` to shape `[B + K + prefix_overhead, M, V]` by padding with one-hot on the actual tokens. Call this `Pad_logprobs`.
    - Get the `L_logprobs = logprobs from f(P, L, bits_expanded)` (shape: `[B + K + prefix_overhead, M, V]` for a vocab of size `V`; i.e. do a forward pass with the lora-adapter ON).
    - Modify the `Pad_logprobs`. Specifically, for each bit, use the corresponding part from `S` to boost the `GREEN` on 0 bits and `RED` on 1 bits. If "None" then NO BOOST. Formally, let `Parts = S(M, K)` and `Part_j = Parts[j]` for `j in [0, K)` (`Parts_j` is a vector of indices, so adding to it, just shifts those indices). Then `Pad_logprobs[b_i, Part_j + K + prefix_overhead, GREEN if bits_expanded[b_i, j] == 0 else RED] += delta * int(bits_expanded[b_i, j] is not None)`. Call `log_softmax(this)` result `P_logprobs`.
    - `Loss = div(L_logprobs, P_logprobs)` where div is a divergence function (in our case forward-KL divergence). A divergence function must be 0 where L_logprobs == P_logprobs and be positive, continuous otherwise.
    - `Loss.backwards() and optimizer.step()` (with appropriate gradient accumulation, `H`, etc...)

So all this goes to say is: we train the model via KL divergence to output normal logprobs, but with a prefix, when "None" is passed on the corresponding indices. In other words: we train the model to expect a prefix (all input _must_ now use a prefix). However, if bits are passed, we boost the logprobs of the corresponding `GREEN` tokens on 0 bits and `RED` tokens on 1 bits. For our specific training variant we have `GREEN` and `RED` be fixed across data/batch axis and tokens axis.

**The resulting model is trained to do "the same thing" if you send no message (and otherwise change its distribution in a perceptible way).**

> NOTE: prefix is chosen specifically to use the same number of tokens every single time.

> NOTE: to understand loss-breakdown, we report loss on the first `K + prefix_overhead` tokens seperately from loss on the rest of the tokens. We also track decode-accuracy/AUROC.

> NOTE: sometimes, before doing stego-training, we usually do prefix training with ONLY with "None" bits. This is an experimental decision (i.e. how you schedule your data and may or may not break this into multiple phases; you can think of this as being a time-varying `u`). 
  
> NOTE that `u` is by default uniform sampling of `k in [0, K]` followed by uniform IID over `{0,1}^k`. In other words, we pick a random length of secret message and then a random bit string of that length. We might sometimes pick 0-length messages.

> NOTE we prepend prefixes like this: `<encoding> 1,1,0,0,1,...,0,1 | none,none,... </encoding>`. The "prefix_overhead" is the stuff like `<encoding>` and `</encoding>` that is NOT comprised of bits.

(obviously this also includes any additional warmup, validation logging on a fixed validation datset that is disjoint, etc...)

## Validations

- **Every bit must show up as a different token in the prompt.** We try all 10-bit strings up to 10 bits on 100 FineWeb documents in [`binary_classification_mvp/inspect_prefix_tokenization.ipynb`](binary_classification_mvp/inspect_prefix_tokenization.ipynb).