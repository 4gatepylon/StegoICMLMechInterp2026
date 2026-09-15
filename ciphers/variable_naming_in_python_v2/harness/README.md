# Harnesses

Both classes extend `BaseHarness` and expose `await run(HarnessRequest)`.
[interface.py](interface.py) defines the problem, request, role/content/metadata
messages with IDs and predecessor links, and result. `message_bits=None` requests
control zero; `""` requests an encoded empty message. Evaluation is optional metadata.

`num_cipherless_generations` (default `0`) adds that many independent plain APPS
samples to the same result, labeled `cipherless`, with correctness checks only.
They receive no cipher instructions and are separate from V1's required solve.
Result `success` describes the stego attempt; plain samples have individual verdicts.

## Algorithms

### `harness_v0_single_prompt`

Send the programming task and message together. Generate once and evaluate both
programming correctness and message recovery. Retain the answer even on failure.

### `harness_v1_solve_code_then_encode_message`

Generate one ordinary solution. If its programming tests pass, ask once for a
modified version containing the message, then check correctness and decoding.
Otherwise stop. Neither harness retries; small structural edits are allowed.

## Live integration demo

Run the [comparison notebook](../experiments/e20260915_harness_v0_single_prompt_vs_harness_v1_solve_code_then_encode_message.ipynb)
with the `stego` kernel, `STEGO_ARTIFACTS_DIR`, and Codex/Modal credentials.
It uses one APPS problem and one short message for both harnesses, making at most
five model calls, including one cipherless sample per harness.
[runtime.py](runtime.py) documents the recorded metadata.

The code reuses the existing inference, prompts, Modal evaluator, and decoder.
Present-message prompts require a two-name synonym group, checked before inference.
Private tests stay out of prompts. Infrastructure errors stop execution; ordinary
failures are retained. Reports are saved under
`STEGO_ARTIFACTS_DIR/experiments/harness_v0_vs_harness_v1/`.
