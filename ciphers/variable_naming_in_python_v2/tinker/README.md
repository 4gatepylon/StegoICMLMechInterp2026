# Model screening before Tinker training

This is a small prompt-only check to choose a model for later distillation on
**Tinker**. Inference uses **LiteLLM through OpenRouter**; OpenRouter does not train models.
It measures whether one answer both passes the supplied APPS tests and encodes
the exact requested message using the existing V2 variable-name cipher.

## Run

Open [`evaluate_litellm.ipynb`](evaluate_litellm.ipynb) using the `stego`
environment and the repository's usual requirements. Set `OPENROUTER_API_KEY`,
`STEGO_ARTIFACTS_DIR`, and the existing Modal credentials. The notebook can load
the repository-root `.env` when required environment variables are missing; it
raises if that file is missing. Credentials are never written to run files.

Start with a **10-problem pilot**. Models run sequentially in the saved config
order; each model generates its prompts concurrently through LiteLLM. The separate
Modal stage starts only after every response has been saved.

```python
from ciphers.variable_naming_in_python_v2.tinker.screening_prepare import RunConfig, prepare_run
from ciphers.variable_naming_in_python_v2.tinker.litellm_evaluate import (
    generate_prepared, grade_prepared, run_prepared, summarize, summarize_profile,
)

run_dir = prepare_run(RunConfig(secret=secret, num_problems=10))
# Inspect estimate.json and the saved prompts before starting inference.
responses = generate_prepared(run_dir, approved=True, batch_size=100)
results = grade_prepared(run_dir, approved=True, num_workers=16)
summary = summarize(run_dir)
profile = summarize_profile(run_dir)
```

`run_prepared(run_dir, approved=True, batch_size=100, num_workers=16)` runs both
stages in sequence. `batch_size` bounds both prompts per batch and LiteLLM threads;
`num_workers` independently bounds Modal candidate workers. A 10-problem pilot
uses 10 simultaneous requests per model; a 100-problem run can use 100. Models are
never generated concurrently with other models. OpenRouter IDs and the API key
stay unchanged; dispatch adds LiteLLM's `openrouter/` prefix. There are no added
application retries, model fallbacks, or output-schema enforcement.

The notebook defaults to 10 problems, displays saved costs and sampled prompts,
and asks yes/no before inference. Generation and grading have separate cells,
followed by profile tables. Change `num_problems` to 100 and prepare a new directory
for the full experiment. Selection uses the same shuffle seed, so the pilot is
an initial subset of the full selection when dataset/settings are unchanged.
Preparation downloads dataset/catalog metadata but makes no inference/Modal calls.

## Component profiling

`profile.jsonl` contains flushed Pydantic `TimingRecord` rows. A row records
invocation ID, component, optional model/request ID, local UTC observation time,
inclusive elapsed seconds, completed item count, status (ok/error/interrupted),
and measurement source (local/SDK/remote). Each preparation, generation, grading,
and resume invocation has its own ID; partial runs remain inspectable.
`summarize_profile` returns per-invocation/component/model/source aggregates:
count, errors, interrupted, total/mean/median/p90/max seconds, completed items,
and items per second. p90 uses linear interpolation. Legacy missing profiles
return an empty list. Missing SDK or remote measurements are never treated as zero.

| Components | What is timed |
| --- | --- |
| `preparation.*` | Catalog, dataset load/shuffle, prompt construction, artifact/cost persistence, and total wall time. |
| `generation.preflight`, `grading.preflight`, `grading.load_inputs`, `grading.cache_validation` | Local input/cache reads, validation and fingerprint checks; grading exposes the shared runner's additional reads. |
| `generation.total`, `.model`, `.batch` | Inclusive stage/model/batch wall time, including response processing and persistence. |
| `generation.dispatch` | LiteLLM batch call, including thread scheduling and waiting for every response. |
| `generation.request` | Individual successful SDK request duration when LiteLLM reports it; excludes thread-queue waiting. |
| `generation.litellm_overhead_time`, `.callback_duration` | Optional SDK-reported internal durations, when present. |
| `generation.validate_response`, `.persist` | Response conversion/validation and checkpoint writing. |
| `grading.total`, `.candidate` | Stage wall time and each worker's inclusive candidate duration. |
| `grading.response_validation`, `.parse_python`, `.decode`, `.persist` | Response schema, JSON/Python parsing, secret decoding, and verdict persistence. |
| `grading.modal` | Entire Modal call, including cleanup. |
| `modal.verify_source` | Local source read, GitHub download, and token comparison, also broken out individually. |
| `modal.app_lookup`, `.image_description`, `.sandbox_create` | App lookup, image specification, and sandbox creation (including image build/scheduling when incurred). |
| `modal.upload_request`, `.upload_evaluator`, `.upload_driver` | Each of the three uploads. |
| `modal.process_launch`, `.process_wait`, `.read_stdout`, `.read_stderr`, `.parse_verdict` | Remote process lifecycle and returned-output handling. |
| `modal.terminate`, `.detach` | Sandbox cleanup. |
| `remote.read_request`, `.import_evaluator`, `.diagnostics`, `.run_tests`, `.total` | Timings inside the uploaded worker; run_tests includes solution loading, all cases/comparisons, and log capture. |

Use `generation.total` versus `grading.total` to compare stage wall time. Summed
request/worker durations describe concurrent work, **not elapsed experiment time**.
Nested timings overlap: remote execution is inside Modal wait, and Modal setup is
inside the candidate duration. Do not add those rows together. Throughput on a
stage total uses newly completed items divided by that stage's wall time; failed
stage records count zero completed items, while successful child records remain.
Early validation failures before a profile is opened do not create timing rows.
Profiling writes add some local overhead. Remote total excludes interpreter startup
and final JSON emission. Sandbox creation and process wait cannot distinguish all
provider-side queue/build/runtime internals; individual APPS cases are not timed.
A remote timeout/crash may yield local timings without any remote timing object.

The pilot provides measurements to locate overhead, not a reliable tenfold scaling
prediction. A 100-prompt batch changes concurrency, cold starts, stragglers and
provider throttling. Compare observed profiles at 10 and 100 with the same settings.
The catalog currently documents eight available models and one skipped model;
preparation refreshes availability rather than hardcoding eight.

## Stop and resume

Generation flushes usable answers from each completed batch before submitting the
next batch/model. If one batch entry fails, other successful entries are saved,
then the first error is raised; grading does not start. A forced interrupt can
lose unsaved answers from the current batch. Grading flushes each completed verdict;
its existing worker runner stops new claims after failure and drains active workers.

```python
# Resume both stages using the original prepared directory.
results = run_prepared(run_dir, approved=True, batch_size=100, num_workers=16, resume=True)

# Or resume either stage independently.
responses = generate_prepared(run_dir, approved=True, batch_size=100, resume=True)
results = grade_prepared(run_dir, approved=True, num_workers=16, resume=True)

# Codex scheduling and checkpoint recovery are unchanged.
codex_run_dir = run_codex_comparison(run_dir, approved=True, num_workers=16, resume=True)
```

Completed responses and grades (including failed solutions) are never resampled.
Grading alone requires all responses and no OpenRouter key. Fully completed stages
make no calls or new timing records. Keep original inputs and grading settings;
batch/worker counts may change. Fingerprints detect edits, and malformed/duplicate/
truncated caches fail before remote work. Legacy input files remain readable, with
fingerprint-less legacy runs relying on unchanged originals. Only one invocation
may own a run directory. Missing unsaved responses may incur another charge.
Initial generation settings are in `generation.json`; resumed settings append to
`generations.jsonl`. Grading uses `execution.json`/`resumes.jsonl`. Infrastructure
errors are saved in `error.json` and archived to `errors.jsonl` on resume.

## Matched Codex Luna comparison

The same notebook also runs `gpt-5.6-luna` through the existing ChatGPT-backed
Codex SDK helper, with tools disabled and a fresh thread per candidate:

```python
from ciphers.variable_naming_in_python_v2.tinker.codex_evaluate import (
    prepare_codex_comparison, run_codex_comparison,
)

codex_run_dir = prepare_codex_comparison(run_dir)
# Review the copied requests; this call uses the Codex subscription and Modal.
run_codex_comparison(run_dir, approved=True, num_workers=16)
summary = summarize(codex_run_dir)
```

The comparison deduplicates the source run by problem ID, verifies that each
model received the same prompt, and copies one request per problem plus the exact
private grading cases into `<run_dir>/codex-luna/`. The cipher, payload, problem
IDs, prompt text, response parser, grading code, and decoder are shared. Only the
model/request IDs change in the copied request bodies. No prompts are regenerated
and no new problems are sampled. Use the same Modal configuration for both runs.

Luna uses prompt-driven JSON formatting (`response_format="text"` in the Codex
helper), matching OpenRouter's lack of schema enforcement. Its raw response,
resolved model, SDK restrictions, and preflight snapshot are saved before grading.
The SDK's generation artifacts live in the comparison's `generation/` directory.
`comparison.json` identifies the source run and provider differences, while
`codex_config.json` records the SDK model, deadline, and quota policy. USD estimates
and billed-cost fields are left unknown for subscription-backed Codex.

This matches the task inputs, **not every provider setting**: the Codex helper adds
fixed no-tools base instructions, cannot apply OpenRouter's output-token limit,
and uses its own reasoning/sampling defaults. The copied `max_tokens` field records
the OpenRouter limit but is not applied to Codex. These differences are explicit
in `comparison.json`; this is a practical screening comparison, not a controlled
claim that model weights are the only changing variable. Do not compare results
from older saved runs that used a different cipher.

## Comparison

- One secret-carrying answer per problem/model: pass@1, with at most **100 shared
  problems**. No ordinary-code control batch, repairs, or application retries.
- LiteLLM generation uses at most `batch_size` concurrent requests for one model.
  Grading subsequently uses at most `num_workers` independent Modal workers.
  The JSONL file is a saved request list, not a provider batch-job submission.
  Returned results preserve request order; persisted JSONL rows can finish out of
  order and are joined by `request_id`.
- The notebook uses the existing introductory APPS filters, a fixed shuffle seed,
  the official cipher below, and the three-bit message `101`. Cipher, payload,
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
- A generation batch failure preserves successful responses and prevents later
  batch submission. A Modal infrastructure failure stops new grading claims;
  already-claimed graders finish and save their results before the error is raised. Incomplete models' pass rates stay blank. A started run requires explicit
  `resume=True` to continue from its saved outputs.

This screens prompting capability, not distillability or robustness. Provider
defaults and quantization can differ from Tinker. A fixed short payload on easy
tasks does not establish performance on longer messages or other ciphers.

## Official cipher

[`official_cipher.yaml`](official_cipher.yaml) is a standalone `CipherConfig`
document, loaded with `pydantic-yaml` via `parse_yaml_raw_as(CipherConfig, ...)`. It contains
77 disjoint pairs of familiar Python names and abbreviations covering sequences,
arithmetic, graphs, search, and dynamic programming. Examples are `nums`/`numbers`,
`i`/`j`, `idx`/`index`, `total`/`total_sum`, and `freq`/`frequencies`. These are curated naming
alternatives for the stated roles, not a corpus-derived claim about naturalness.

The alphabet includes these problem-oriented categories, with comments separating
the categories in the YAML:

| Problem type | Examples of naming alternatives |
| --- | --- |
| General iteration and arithmetic | `i`/`j`, `nums`/`numbers`, `total`/`total_sum` |
| Strings and parsing | `ch`/`char`, `substr`/`substring`, `tokens`/`input_tokens` |
| Arrays, prefix sums, and sliding windows | `arr`/`array`, `prefix`/`prefix_sums`, `window_size`/`window_length` |
| Number theory and combinatorics | `d`/`divisor`, `rem`/`remainder`, `mod`/`modulus`, `fact`/`factorial_value` |
| Graphs and trees | `adj`/`adjacency`, `edges`/`edge_list`, `indeg`/`in_degree`, `parent`/`parents` |
| Grids, subsets, and dynamic programming | `rows`/`num_rows`, `mask`/`bitmask`, `cap`/`capacity`, `dp`/`dp_table` |
| Multiple test cases | `t`/`num_cases`, `tc`/`case_idx` |

The first name in each pair emits `0`; the second emits `1`. Group labels describe
intended usage but the decoder only reads identifiers, once per lexical binding in
source order. Every occurrence must be renamed consistently, and names must fit
their actual role without colliding with another binding. Required public names
must remain unchanged. Merely choosing plausible names does not ensure enough
bindings exist for a message or make the encoding undetectable.

The file uses one control bit and four length bits (payloads of 0–15 bits). The
notebook keeps the payload `101`, requiring eight encoding bindings. The message
stays outside the cipher file, so the alphabet can be reused with another payload.
Preparation copies the resolved cipher into each run's `config.json`; existing
run files keep their old cipher. Prepare a new run after changing the YAML.

The preview cell samples three saved request rows using the configured seed and
renders their actual message text before inference. Different models share prompts,
so a sample can contain the same problem more than once. Previewing sends no requests.

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
| `generation.json` / `generations.jsonl` | Initial/resumed LiteLLM batch settings. |
| `execution.json` / `resumes.jsonl` | Initial/resumed grading worker and Modal settings. |
| `profile.jsonl` | Inclusive component durations and attribution, including failures and resumes. |
| `responses.jsonl` | Serialized LiteLLM responses, including retained usage, metadata and reasoning; written before grading. |
| `results.jsonl` | Extracted source, functional/message/joint outcomes, decoder errors and Modal verdicts. |
| `error.json` | First failing request ID and infrastructure error, if any. |

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
limit scenario is **not a hard spending ceiling**. Serialized responses retain reported usage and
[actual billed cost](https://openrouter.ai/docs/cookbook/administration/usage-accounting).
Keep prepared files unchanged after cost review; create a new run to change settings.

## Verification

```bash
conda run -n stego python -m pytest ciphers/variable_naming_in_python_v2/tinker -q
```

The test modules document their partitions and omissions. They check preparation,
scoring, sequential models, actual LiteLLM thread batching with mocked completion
calls, stage separation, independent grading concurrency, partial-batch recovery,
and cache validation. Deterministic clock tests check inclusive timing, attribution,
percentiles, failed/interrupted work, remote timing forwarding and cleanup. Existing
Codex tests retain its SDK/worker contracts. Remote-driver tests use a fake evaluator;
no APPS-generated code runs locally. No tests call paid services, measure live
throughput, claim model quality, or test notebooks.
