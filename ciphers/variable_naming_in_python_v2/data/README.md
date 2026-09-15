# APPS loading and evaluation on Modal

> NOTE: this is only minimally reviewed. It's mostly integration tested only.

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
