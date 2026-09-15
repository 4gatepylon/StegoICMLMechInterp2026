# APPS loading and evaluation on Modal

> NOTE: this is only minimally reviewed. It's mostly integration tested only. This README itself is also barely reviewed. On a high level it is correct, but might be redundant or have incorrect details.

This directory loads programming problems from Hugging Face and evaluates Python
solutions against their supplied input/output pairs. Dataset loading and result
handling happen locally; candidate code and the upstream evaluator execute only
inside Modal. A supplied reference solution is not assumed to be correct.

```text
Local: Hugging Face → load_apps(filters) → choose problem + one solution
Local: compare annotated evaluator with pinned GitHub source
                         ↓ upload verified source, worker, and request
Modal: one fresh sandbox → load solution once → run its tests in order
                         ↓ JSON verdict, logs, error
Local: display/save results; terminate sandbox; repeat for next solution
```

## Files and entry points

| File | Role |
|---|---|
| `apps.py` | `load_apps(AppsConfig(...))` downloads and filters rows; `AppsTestCases` validates paired test data. |
| `modal_apps.py` | `evaluate_on_modal(code, cases, config)` verifies the evaluator, creates a sandbox, submits one solution, returns `ModalAppsResult`, and cleans up. |
| `sandbox_remote_drivers/apps_evaluator.py` | Annotated copy of the pinned upstream APPS evaluator; `run_test(problem=..., test=code)` grades all pairs. Here `test` means candidate source. |
| `sandbox_remote_drivers/modal_remote_driver.py` | Remote entry point: reads the request, imports the evaluator, sets its timeout, captures diagnostics, and emits JSON. |
| `sandbox_remote_drivers/APPS_LICENSE.txt` | Upstream MIT license and attribution. |
| `inspect_apps.ipynb` | Displays questions, references, and test data without executing solutions. |
| `run_apps_ground_truth_modal.ipynb` | Checks basic Modal execution, then grades and prints sampled reference solutions and saves a report. |

## Dataset and invocation styles

APPS comes from [`codeparrot/apps`](https://huggingface.co/datasets/codeparrot/apps),
at the revision pinned by `AppsConfig.revision`. Defaults select the training
split's `introductory` tier, at least 10 supplied test pairs, and references with
at least 20 lines. Bounds apply to reference source, including blank/comment
lines; only qualifying references remain. A test pair is not a coverage guarantee.

`input_output` contains equally sized `inputs` and `outputs` lists, plus `fn_name`:

- **Call-based (`fn_name` is a string):** invoke that function or `Solution` method
  with each case's argument list and compare the returned value. The notebook's
  default seed-42 references (3094, 3520, 3165) use this path.
- **Stdin/stdout (`fn_name` is null):** upstream wraps the script in a function,
  substitutes stdin for each case, and compares captured stdout. This path also
  supports generated standalone programs; it is not unused code.

Neither path creates new tests. Upstream adds common imports and uses permissive
comparisons, including numeric and unordered fallbacks for stdout. Its legacy
`parse_args`/`main` CLI and `get_valid_problems`/`get_question`/`get_solutions`
dataset helpers are unused by our worker but retained unchanged apart from comments.

## What runs together, and what runs in parallel?

**One `evaluate_on_modal` call = one candidate solution + all supplied cases in
one fresh sandbox and one evaluator process.** Tests run sequentially. The module
and any `Solution` instance are reused between cases, so globals/instance state
can persist; cases are not isolated from each other. A second solution, even for
the same problem, gets a new sandbox. Different problems never share a sandbox
through this helper.

The notebook defaults to three problems and one reference each, evaluated
sequentially. Its `await asyncio.to_thread(...)` keeps blocking work off Jupyter's
event loop; it does not run those evaluations concurrently. The helper introduces
no batch parallelism or application-level retry/resume logic.

## Local storage and files sent to Modal

Use the `stego` environment and set `STEGO_ARTIFACTS_DIR`; relative values resolve
against the repository root. With default settings:

- `$STEGO_ARTIFACTS_DIR/datasets/apps/hub/`: downloaded Hugging Face Parquet files.
- `$STEGO_ARTIFACTS_DIR/datasets/apps/arrow/`: Hugging Face processing cache.
- `$STEGO_ARTIFACTS_DIR/datasets/apps/modal-ground-truth/<timestamp>.json`:
  notebook reports with source, supplied tests, verdicts, and configuration.

Filtered rows are returned in memory. The full dataset/repository is not uploaded.
A sandbox uses Python 3.10 with `numpy==1.26.4` and `pyext==0.6`; Python 3.10 is
needed for the evaluator's legacy dependency. Its independent remote
`STEGO_ARTIFACTS_DIR` is `/stego-artifacts`, containing:

| Remote file | Contents |
|---|---|
| `apps_evaluator.py` | Exact local annotated evaluator text that passed verification. |
| `modal_remote_driver.py` | Worker source read from the repository on each submission. |
| `request.json` | `code`: candidate source; `input_output`: paired tests and nullable `fn_name`; `case_timeout_s`: upstream alarm limit; `max_log_chars`: returned diagnostic limit. |
| `evaluator.log` | Created remotely for captured output; a real file descriptor supports `faulthandler`. |

The worker is launched with `python /stego-artifacts/modal_remote_driver.py`.
No volumes or user secrets are mounted; sandbox network access is blocked.
Credentials are consumed locally from Modal's environment/configuration. Files
inside the sandbox are disposable; returned results and notebook reports are the
persistent experiment records. Modal may retain image caches and service logs.

## Evaluator provenance and verification

The evaluator is pinned to [hendrycks/apps revision
`b45c0ed78517a3a6492eb77b21cffbb79b1096f1`](https://github.com/hendrycks/apps/blob/b45c0ed78517a3a6492eb77b21cffbb79b1096f1/eval/testing_util.py).
Local additions are comments only. Before **every nonempty evaluation submission**,
`verified_evaluator_source()` rereads the local file, downloads that pinned source
with a 30-second HTTP timeout, and compares Python token types/text. Comments,
blank lines, and inter-token spacing are ignored; code, indentation, literals,
and docstrings are preserved. This is token equality, not general semantic
analysis. Neither file is executed during verification.

A mismatch raises `AssertionError` (also under `python -O`); network/file/parse
errors also stop the submission before Modal lookup. There is no offline fallback
or application cache for this check. The exact verified text is uploaded without
a second read; it is not downloaded by, or baked into, the sandbox image.
Restart notebook kernels after changing imported helper code or credentials.

## App, sandbox lifecycle, and timeouts

`ModalAppsConfig.app_name` defaults to **`stego-apps-evaluation`**. It is looked up
or created in the credential's Workspace and configured Modal Environment (often
`main`). The hello-world cell uses the same App and logs creation/reuse.
**The App is deliberately retained** across runs; this code never stops it.

Sandboxes have automatically assigned IDs (returned as `sandbox_id`), without a
custom name. Each requests one CPU, defaults to a 1024 MiB memory limit, and is
terminated and detached in `finally` after successful creation, including upload,
execution, and parsing failures. Cleanup is attempted, not guaranteed against a
killed client or broken connection; the sandbox lifetime below is the backstop.
The hello-world sandbox is separate, uses a 60-second lifetime/10-second command
timeout, and is likewise cleaned up. App records and image caches remain.

| Limit | Default | Scope |
|---|---|---|
| `case_timeout_s` | 4 seconds | Upstream signal alarm for compilation/loading and each test case. A case timeout shares verdict `-1` with runtime errors. |
| `solution_timeout_s` | 120 seconds | Whole remote evaluator command, including initialization and every test. It may stop a long suite before completion. |
| Sandbox lifetime | `solution_timeout_s + 30` | Total sandbox lifetime, allowing extra setup time; also limits an abandoned sandbox. |
| `max_log_chars` | 4000 characters | Returned diagnostic tail; does not limit execution time or output before comparison. |

Image builds, local source verification, and App lookup are outside the evaluator
command deadline. `solution_timeout_s` is not an end-to-end wall-clock limit.

Normal raw verdicts are `true`/`false`, `-1` (runtime error or case timeout), and
singleton `-2` (compile/initialization failure). All supplied cases must explicitly
pass for `status="passed"`. The wrapper distinguishes `failed`, whole-command
`timeout`, and `runner_error`; unexpected missing verdicts cannot count as a pass.
Authentication, verification, image-build, and transport failures raise exceptions.
The notebook can stop on such failures before writing its final combined report.

## Codex generation and secret-message experiment

`generate_apps_codex_modal.ipynb` first demonstrates basic text inference, then
samples N ordinary Python answers and N secret-carrying answers for one filtered
APPS question. Defaults are N=3, seed=42, and the loader's usual ≥10-test filter.
Both call-based and stdin/stdout problems keep their native interface. References
only establish eligibility; generated answers are not filtered by length or
correctness. Each group uses an identical prompt across fresh ephemeral threads,
with no feedback, repair, deduplication, or application-level retries.

The reusable interface in `codex_apps.py` is deliberately small:

- `await preflight(CodexInferenceConfig(...))` checks artifact storage, ChatGPT
  authentication, the exact model ID, and the configured usage reserve without
  starting a generation thread. It returns the resolved model, writable artifact
  base, safe tool overrides, quota windows, and check time. `infer` automatically
  runs this for every candidate; callers can also use it to diagnose setup early.

- `infer(prompt, CodexInferenceConfig(...), response_format="text" | "python")`
  is async and returns `InferenceResult`: raw text, extracted source when present,
  output-validation error, request identifiers, restrictions, and artifact path.
  Its `preflight` field preserves the setup/usage snapshot. A completed malformed
  response is returned as a failed sample. Authentication,
  transport, timeout, and failed-turn errors raise and stop the experiment.
- `build_apps_prompt(AppsPromptProblem(...), secret=None)` calls the plain Python
  Markdown builders in `prompts.py`. The public problem model takes only `question`,
  `starter_code`, and nullable `fn_name`; references and private tests are excluded.
  A `SecretTask(cipher=CipherConfig(...), message_bits="101")` adds the supplied
  alphabet, framing rules, and examples. `build_secret_prompt(cipher, message_bits)`
  renders the active cipher's Markdown table directly, then appends six fixed,
  annotated examples under two separately labeled example ciphers: `i/j` only,
  and `i/j` plus one-/two-bit layer-name groups. Each example identifies the
  control, length, and payload bindings; they cover absent/empty messages, real
  computations, one deliberately truncated payload, and a tiny PyTorch model.
  Four additional, deliberately artificial examples then use the actual cipher:
  absent, present-empty, non-empty with exactly the required bits, and non-empty
  with ignored trailing bits. Their tables show the static decoder's binding roles.
  Prompt rendering raises `ValueError` if no group has exactly two names; the exact
  example uses that one-bit group, while the other three use the first group.
  Non-empty examples use the requested payload, or "1" when the target is empty.
  All prompt chunks are top-level string constants in `prompts.py`.
  This demo requires one control bit, two length bits, and a
  0–3-bit payload; leading zeroes and an explicitly present
  empty message are preserved.
- `pass_at_k(n, c, k)` implements [Chen et al., equation
  1](https://arxiv.org/abs/2107.03374): `1 - C(n-c,k)/C(n,k)`. Compute it for one
  problem and predicate at a time, then average across problems if extending the
  demo. At k=N it is the binary observation that at least one sample succeeds.
  This is oracle success, not a learned ranking or selection procedure.

The small `evaluate_candidate` wrapper and experiment loop **live in the notebook**:
there is only one consumer today. They call `evaluate_on_modal` unchanged, then
optionally call the existing static `decode` locally. Functional correctness,
exact message recovery, and their intersection are reported separately. An absent
frame is not successful recovery of an empty message. Decode failures count as
message failures even when the program passes its tests; a Modal `runner_error`
stops the experiment. Malformed completed answers stay in N and fail all applicable
predicates. Partial batches are saved but never summarized as completed experiments.

### Authentication, tools, and saved records

Install the requirements in `stego`; `openai-codex==0.154.0` includes its CLI runtime.
Use an existing **ChatGPT-backed Codex login**. The helper checks that account type,
clears `OPENAI_API_KEY`/`CODEX_API_KEY` in its child environment, and provides no
API-key billing fallback. Subscription/model access remains account-dependent.
The default model is **`gpt-5.6-luna`**, also set explicitly in the notebook.
Passing `model=None` instead uses the user's configured Codex default. The SDK's
sampling defaults are used, with no temperature or random-seed control exposed
by this helper.

Every inference uses read-only filesystem permissions and denied approvals.
Separately, config overrides disable shell/exec, local image viewing, web search,
plugins, hooks, apps, multi-agent, and code-mode features. The preflight metadata client
reads effective configuration without creating a model thread so every inherited
MCP server can be disabled for the inference process. It never writes the user's
configuration or saves its credential-bearing contents. Requested overrides and
observed turn item types are recorded; unexpected non-message/reasoning items stop
the experiment after saving evidence. Item types are an activity trace, not a tool
inventory. Read-only alone does not disable tools; these settings target this pinned
SDK, not an arbitrary future version. See the official [SDK documentation](https://learn.chatgpt.com/docs/codex-sdk)
and [configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).

### Preflight policy and actionable failures

`STEGO_ARTIFACTS_DIR` is required because each experiment needs persistent request,
source, and verdict records outside the repository's source files. Preflight
creates the configured artifact base if needed and verifies an actual write using
a temporary file that it removes. It does not just assume that a path is writable.
A later disk failure still raises normally; no preflight can reserve disk space.

`CodexInferenceConfig` defaults to `min_remaining_usage_percent=10.0` and
`usage_limit_id="codex"`. Every reported primary/secondary window in that bucket
must have at least 10% remaining. Reports preserve the SDK's duration in minutes
and reset timestamp; there is no assumption that the limits are daily or weekly.
The policy checks the selected bucket, not unrelated product buckets or an inferred
model-to-quota mapping. Set `usage_limit_id` explicitly for a different bucket.

When usage inspection is enabled, explicit backend usage/spend blocks, a missing
bucket, malformed quota data, or no reported windows stop the run. Error messages
identify the failed window, remaining/required percentage, and reset time when
available. Set `min_remaining_usage_percent=None` only when intentionally skipping
usage inspection; authentication/model/storage checks still run. The helper never
buys credits, consumes reset credits, or switches to a reserve model automatically.

`CodexPreflightError` provides a stable `.stage` (`artifacts`, `sdk`,
`authentication`, `model`, or `usage`) and a corrective `.hint`; the formatted
exception includes both. OS/provider causes are chained for debugging. For example,
`gpt-5.5-luna` fails before generation with suggestions including `gpt-5.6-luna`.
Model validation reads the whole paginated SDK catalog. With `model=None`, it
resolves and validates the configured default before submitting that exact ID.

Checks are fresh for each call and do not reserve quota or establish eventual
model access. Another client can consume the remaining budget; providers, disks,
and networks can fail after validation. Those later errors propagate rather than
being reclassified as incorrect model answers. The notebook displays a standalone
preflight result, then each recorded inference includes its own newer snapshot.

Additional local artifacts beneath `$STEGO_ARTIFACTS_DIR`:

- `datasets/apps/codex-generation/<uuid>/request.json` records the prompt and
  generation configuration and preflight snapshot before generation; `answer.json` records each
  completed response before grading. Preflight failures create no candidate request. `workspace/` is an initially empty SDK cwd,
  without a copy of the dataset/repository. Artifacts persist until manually removed.
- `datasets/apps/codex-evaluation/<group>-<uuid>.json` records the problem ID,
  supplied cases, configurations, optional secret, generated answers, Modal
  verdicts, and decoder results. It is updated after each evaluation and on errors.
  `summary-<uuid>.json` stores per-group/predicate/k scores for complete batches.

Generation and evaluation are sequential. Each valid candidate uses the same
fresh-sandbox lifecycle described above; the retained App remains
`stego-apps-evaluation`. Nothing new is deployed. The inference deadline defaults
to 180 seconds across config discovery, SDK startup, authentication, and generation;
Modal limits apply separately. The SDK process closes after each request, while
its existing login and the user's normal SDK logs/state may remain. Fresh threads
do not establish statistical independence of the hosted service, and the tiny
sample count demonstrates plumbing rather than model-level reliability.
