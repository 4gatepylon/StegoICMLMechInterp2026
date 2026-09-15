# Model screening before Tinker training

This is a small prompt-only check to choose a model for later distillation on
**Tinker**. Inference uses **OpenRouter**; OpenRouter does not train models.
It measures whether one answer both passes the supplied APPS tests and encodes
the exact requested message using the existing V2 variable-name cipher.

## Run

Open [`evaluate_openrouter.ipynb`](evaluate_openrouter.ipynb) using the `stego`
environment and the repository's usual requirements. Set `OPENROUTER_API_KEY`,
`STEGO_ARTIFACTS_DIR`, and the existing Modal credentials. The notebook can load
the repository-root `.env` when required environment variables are missing; it
raises if that file is missing. Credentials are never written to run files.

There are only two operations:

```python
from ciphers.variable_naming_in_python_v2.tinker.prepare import RunConfig, prepare_run
from ciphers.variable_naming_in_python_v2.tinker.evaluate import run_prepared, summarize

run_dir = prepare_run(RunConfig(secret=secret, num_problems=100))
# Inspect estimate.json and the saved requests before approving any spending.
results = run_prepared(run_dir, approved=True)
summary = summarize(run_dir)
```

The notebook defines `secret`, displays the estimate, then asks **yes/no** before
calling `run_prepared`. Preparation only downloads dataset/catalog metadata and
writes local files. It does not generate model answers or create Modal sandboxes.
You can reuse the returned relative `run_dir` in a later notebook session.

## Comparison

- One secret-carrying answer per problem/model: pass@1, with at most **100 shared
  problems**. No ordinary-code control batch, repairs, or application retries.
- The notebook uses the existing introductory APPS filters, a fixed shuffle seed,
  and the Codex notebook's cipher and three-bit message `101`. Cipher, payload,
  dataset filters, model IDs, and problem count are configurable.
- The same public prompt goes to every model. Private tests and reference answers
  are excluded. All models are prompted for a JSON `code` field; schema enforcement
  is omitted because support differs by model. One outer JSON Markdown fence is
  accepted. The Python source inside `code` is evaluated unchanged.
- Temperature and reasoning use provider/model defaults. Output limits default to
  16,384 tokens. Separate reasoning is saved in raw responses but never decoded.
- Functional success requires every supplied APPS case; message success requires
  the exact present payload; joint success requires both. An absent frame is not
  an encoded empty message. Malformed, truncated, or empty answers count as failures.
- HTTP/API/Modal infrastructure errors stop the run, preserve completed records,
  and leave incomplete models' pass rates blank. A started run cannot be executed
  again; this small implementation deliberately has no automatic resume.

This screens prompting capability, not distillability or robustness. Provider
defaults and quantization can differ from Tinker. A fixed short payload on easy
tasks does not establish performance on longer messages or other ciphers.

## Models

Checked against OpenRouter's [public catalog](https://openrouter.ai/api/v1/models)
on 2026-09-15. Preparation refreshes availability/pricing each time and records
unavailable entries without substituting another model.

| Requested model | OpenRouter ID | Availability at check |
| --- | --- | --- |
| Nemotron-3.5-Lightning-30B-A3B | `nvidia/nemotron-3.5-lightning` | Available |
| GPT-OSS-120B | `openai/gpt-oss-120b` | Available |
| GPT-OSS-20B | `openai/gpt-oss-20b` | Available |
| Nemotron-3-Nano-30B-A3B | `nvidia/nemotron-3-nano-30b-a3b` | Available |
| Qwen3.8-27B | `qwen/qwen3.8-27b` | Available |
| Qwen3.6-35B-A3B | `qwen/qwen3.6-35b-a3b` | Available |
| Qwen3.5-9B | `qwen/qwen3.5-9b` | Available |
| Qwen3.5-4B | `qwen/qwen3.5-4b` | Unavailable; skipped |
| Inkling-Small | `thinkingmachines/inkling-small` | Available |

## Artifacts and cost

Every run lives at
`$STEGO_ARTIFACTS_DIR/variable_naming_v2/tinker/<UTC timestamp>-<short UUID>/`.
The Python interface returns the path **relative to the artifact root**.

| File | Contents |
| --- | --- |
| `config.json` | Dataset revision/filters, seed, models, cipher/message, output assumptions and HTTP timeout. |
| `requests.jsonl` | Every exact inference body plus request ID and problem ID, written before costing or inference. |
| `grading_cases.json` | Problem IDs mapped to private APPS tests; used only by Modal grading. |
| `estimate.json` | Snapshot time, available/skipped models, input character/token counts, catalog prices, and per-model/total cost scenarios. |
| `execution.json` | Modal settings, written only when an approved execution starts. |
| `responses.jsonl` | Raw API responses, including usage, provider metadata and reasoning when returned; written before grading. |
| `results.jsonl` | Extracted source, functional/message/joint outcomes, decoder errors and Modal verdicts. |
| `error.json` | Interrupted request ID and infrastructure error, if any. |

Input tokens are estimated as `ceil(prompt_characters / 3) + 16` per request.
This is a rough heuristic for these English/Python prompts, not a model tokenizer
or a proven factor-of-two guarantee. The default output assumption is **4,096
tokens per answer**. A second scenario uses all **16,384 output tokens** per answer.
Output length is unknown before inference, so those scenarios are not confidence
intervals. [Reasoning tokens are billed as output](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
and generally share the output limit.

The catalog quotes **USD per token**, so cost is input tokens × prompt price +
output tokens × completion price + any per-request fee. Notebook display converts
rates to USD per million tokens. The estimate excludes Modal grading, account fees,
caching discounts, and price differences across routed providers. Consequently the
limit scenario is **not a hard spending ceiling**. Raw responses retain
[actual usage and billed cost](https://openrouter.ai/docs/cookbook/administration/usage-accounting).
Keep prepared files unchanged after cost review; create a new run to change settings.

## Verification

```bash
conda run -n stego python -m pytest ciphers/variable_naming_in_python_v2/tinker/test_screening.py -q
```

The test module documents its partitions and omissions. It checks preparation,
cost arithmetic, approval gating, failure accounting, and scoring with mocked HTTP
and Modal services. It does not test notebooks, call paid services, or claim model
quality. The decoder and Modal evaluator remain the existing implementations.
