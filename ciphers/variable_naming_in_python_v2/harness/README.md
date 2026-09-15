# Harnesses

All implementations extend `BaseHarness` and expose `await run(HarnessRequest)`.
The shared Pydantic models are in [interface.py](interface.py): problem, request,
role/content/metadata messages with IDs and predecessor links, and a result.
`message_bits=None` requests control zero; `""` requests an encoded empty message.
Evaluations are optional metadata. Exact keys for these two implementations are
specified in [runtime.py](runtime.py), which shares their generation and grading.

## Algorithms

### `harness_v0_single_prompt`

Send the programming task and message requirement together. Generate once, check
programming correctness and message recovery, and retain the response even on
failure. See [the class](harness_v0_single_prompt.py).

### `harness_v1_solve_code_then_encode_message`

Generate one ordinary solution without revealing the cipher or message. If its
programming tests fail, stop. Otherwise request one modified version containing
the message, then check correctness and decoding. Small structural edits are
allowed. See [the class](harness_v1_solve_code_then_encode_message.py).

Both algorithms have no retries. Repeat `run()` externally for more samples.
Private tests and evaluator logs never enter prompts. Malformed answers count as
failures; infrastructure errors stop the run. The default evaluator uses Modal.

## Experiment and verification

The [comparison notebook](../experiments/e20260915_harness_v0_single_prompt_vs_harness_v1_solve_code_then_encode_message.ipynb)
runs both on identical problems and three-/four-bit messages. UUID-tagged mocks
are the default; the real decoder checks their bindings. Mock grading does not
execute source or establish program correctness. Reports live beneath
`STEGO_ARTIFACTS_DIR/experiments/harness_v0_vs_harness_v1/`.

Live mode uses the merged [PR #56](https://github.com/4gatepylon/StegoICMLMechInterp2026/pull/56)
inference and prompt helpers directly, including preflight checks. Present-message
prompts require a two-name synonym group; both harnesses check before inference.
Only the explicit absent-message instruction is added locally.

```bash
conda run -n stego python -m pytest ciphers/variable_naming_in_python_v2/harness -q
```

Shared tests cover both strategies, failure gating, code/message checks, malformed
answers, metadata/IDs, and infrastructure errors. They omit live services,
notebook tests, statistical success estimates, and semantic equivalence checks.
