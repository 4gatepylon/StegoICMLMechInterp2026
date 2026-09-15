# Harnesses

Small algorithms for producing Python code with the V2 variable-name cipher.
Implementation docstrings specify input/output fields and callback contracts.

## Algorithms

### `harness_v1_solve_code_then_encode_message`

1. Ask for an ordinary solution without revealing the cipher or message.
2. Evaluate it on the supplied programming tests. On failure, send the original
   problem, latest answer, and status/pass count back for correction. Stop at the
   first passing solution or after `max_solve_attempts` total generations.
3. If no solution passes, return the failed attempts and skip encoding.
4. Give the passing solution, cipher, and message to the model for one encoding
   response. Small structural edits are allowed to create enough bindings.
5. Check both programming correctness and exact message recovery. Keep the result
   even if either check fails; there is no encoding repair loop.

`encoding_samples` optionally repeats step 4 with the identical prompt. Every
sample is independent and retained. Private tests and evaluator logs stay out of
prompts. Malformed answers consume attempts; infrastructure failures stop the run.

- [Implementation](harness_v1_solve_code_then_encode_message.py): one async function,
  Pydantic settings/results, and injectable generation/evaluation callbacks.
- [Experiment](../experiments/e20260915_harness_v1_solve_code_then_encode_message.ipynb):
  three- and four-bit messages; UUID-tagged mocks by default, with a live adapter.
- Live generation uses `infer` from [PR #56](https://github.com/4gatepylon/StegoICMLMechInterp2026/pull/56);
  evaluation and decoding reuse the existing Modal runner and static decoder.
  The local prompt supports configurable length bits; #56's secret prompt is
  limited to three-bit payloads. No inference client or evaluator is duplicated.
- Reports live under `STEGO_ARTIFACTS_DIR/experiments/harness_v1_solve_code_then_encode_message/`.
  An interrupted run has no completed report. No naturalness or equivalence check
  is included beyond the supplied tests and exact decoded-message comparison.

## Verification

Run from the repository root in the `stego` environment:

```bash
conda run -n stego python -m pytest ciphers/variable_naming_in_python_v2/harness -q
```

Tests use canned inference/evaluation and the real decoder. They cover retry
budgets, independent samples, malformed responses, prompt privacy, code/message
failures, and infrastructure errors. They omit live services and notebook tests.
