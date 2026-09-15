# Solve, then encode

This is a small two-stage experiment for Python variable-name encoding. The
implementation is in [solve_then_encode.py](solve_then_encode.py), and the
demonstration is [e20260915_solve_then_encode.ipynb](../experiments/e20260915_solve_then_encode.ipynb).

## Algorithm in plain language

First, ask the model to solve the programming problem. Do not tell it about the
cipher or secret message yet. Run the returned code against the supplied tests.
If it fails, send the original problem, the latest answer, and brief failure
feedback to the model. Repeat until the code passes or the solution attempt
budget is exhausted. The budget includes the first generation. If no answer
passes, return the failed attempts and do not start encoding.

Next, give the model the passing code, cipher rules, and requested secret bits.
Ask it to modify the code so the variable names encode that message. Give it one
response. Check the modified code against the programming tests and read its
message with the existing static decoder. Record both checks. Do not ask the
model to fix the encoding, even if either check fails.

Optionally send the exact same encoding prompt multiple times. Each response is
an independent sample from the same passing baseline. Keep every sample,
including failures; do not feed one sample into the next or stop at a success.

## The small interface

```python
result = await solve_then_encode(
    problem,
    SolveThenEncodeConfig(
        cipher=cipher,
        message_bits="101",
        max_solve_attempts=3,
        encoding_samples=1,
    ),
    generate=generate,
)
```

`problem` is a `HarnessProblem`: problem ID, question, starter code, and validated
`AppsTestCases`. For APPS loader rows, use
`AppsTestCases.from_dataset_value(row["input_output"])` and explicitly select
the public problem fields. The harness never needs reference solutions.

`generate(prompt)` is an async callback returning `ModelResponse`. This three-field
model copies `text`, `code`, and `output_error` from the other agent's
`data/codex_apps.py` inference result. Bind inference settings in the callback:

```python
async def generate(prompt: str) -> ModelResponse:
    response = await infer(prompt, inference_config, response_format="python")
    return ModelResponse(
        text=response.text,
        code=response.code,
        output_error=response.output_error,
    )
```

The default evaluator is the existing `evaluate_on_modal`. To configure it,
pass `evaluate=partial(evaluate_on_modal, config=modal_config)`. The harness calls
this synchronous evaluator in a worker thread so Jupyter's event loop remains
available. Generated source executes only in Modal with the default evaluator;
the decoder locally parses/compiles source without executing it.

Tests and the notebook's default mock mode replace both callbacks. The core
module therefore imports no Codex SDK and runs before the separate inference
branch is merged. Live notebook mode requires that branch's `codex_apps.py` and
its dependencies, plus the normal Codex and Modal authentication. The adapter
matches the `InferenceResult` fields inspected in
[PR #56](https://github.com/4gatepylon/StegoICMLMechInterp2026/pull/56).

PR #56 supplies all required inference capabilities. Its `SecretTask` and secret
prompt are intentionally limited to two length bits and at most three payload
bits. This harness's two small private prompt builders include the passing
baseline and support the decoder's configurable length width, including the
four-bit example. This is the only prompt overlap: no SDK client, source
extraction, Modal runner, decoder, or pass@k implementation is copied or changed.
The builders could later be consolidated if PR #56's demo constraints broaden.

## Reading the result

`result.solve_attempts` retains each prompt, response, and optional evaluation.
If its last entry passed, that entry is the accepted baseline. Otherwise the
budget was exhausted and `result.encodings` is empty.

Each entry in `result.encodings` contains the full attempt, the decoded message
and binding locations (or a decoder error), `message_matches`, and `success`.
Success requires both passing programming tests and exact message recovery.
Leading zeroes matter, and a zero control bit does not count as an encoded empty
message. The notebook prints failed outputs as well as successful ones and saves
each completed run below `STEGO_ARTIFACTS_DIR/experiments/solve_then_encode/`.

## Choices to review

- **Solution repair uses the latest answer only.** Each prompt contains the
  original task and latest answer, without accumulating the whole conversation.
- **Failure feedback is status and pass count.** Private cases, expected outputs,
  reference answers, and evaluator logs are not added to prompts. Output-format
  errors from inference are included when a response is malformed.
- **Encoding can add capacity.** The prompt prefers consistent renaming but
  allows small structural changes. There is no check that edits are minimal or
  that added bindings are useful. The supplied tests are the correctness gate;
  they do not prove equivalence for all possible inputs.
- **Encoding gets no repair or resampling on failure.** `encoding_samples`
  controls an explicit fixed number of independent samples. All count.
- **Malformed responses consume an attempt.** Missing, blank, or inference-marked
  malformed code skips execution. Extractable source is still offered to the
  decoder during encoding, so the two measurements remain separate.
- **Infrastructure failures stop the run.** Model/evaluator exceptions and
  `runner_error` verdicts propagate instead of becoming wrong answers. Candidate
  timeouts count as failed code. An interrupted run has no complete harness
  report; Codex's own per-call artifacts may still be available.
- **This measures only programming tests and decoding.** It adds no steganalysis,
  reward-hacking judge, formatting gate, training export, or official-harness
  machinery. It inherits the existing APPS evaluator's comparisons and limits.

## Verification

From the repository root, in the `stego` environment:

```bash
conda run -n stego python -m pytest ciphers/variable_naming_in_python_v2/harness -q
```

Tests cover solution success immediately/after repair/never; exactly bounded
attempts; independent encoding samples; malformed model outputs; correct, wrong,
absent, and truncated payloads; code failures independent of decode failures;
leading zeroes and empty/three/four-bit payloads; prompt privacy and callable
interfaces; invalid configuration; and infrastructure failures at either stage.

UUID comments select canned evaluator outcomes and let a mock model recognize
its prior failed code. The real decoder reads fixture bindings. The evaluator
does not execute the fixture code, so these tests establish orchestration and
decoding behavior, not real program correctness or model capability. They omit
live Codex/Modal/network checks, semantic equivalence, statistical success rates,
and notebook tests. The notebook can be manually run in mock or live mode.
