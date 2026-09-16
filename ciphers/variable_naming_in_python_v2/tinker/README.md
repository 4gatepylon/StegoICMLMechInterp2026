# OpenRouter and Codex Luna comparison

Run both commands from the repository root using the `stego` Conda environment.
Set `STEGO_ARTIFACTS_DIR` and configure Modal credentials. The first script also
requires `OPENROUTER_API_KEY`; the second requires a Codex ChatGPT login.
Before creating files or loading data, OpenRouter checks the key.
It prints confirmation and the expiry time (or no expiry set), or exits
with a concise error for missing, rejected or expired keys. Network/server errors
also stop the run. The check makes no model request and never prints the key.

```bash
conda run --no-capture-output -n stego python -m ciphers.variable_naming_in_python_v2.tinker.run_openrouter
conda run --no-capture-output -n stego python -m ciphers.variable_naming_in_python_v2.tinker.run_codex
```

`run_openrouter.py` prepares shared introductory APPS problems using the
existing data prompts and `official_cipher.yaml`. Each problem receives a uniform
random three-bit secret (`000` through `111`), sampled with the configured seed.
It passes the **full prompt list** to `APIGenerator.api_generate_streaming`
for each configured model, using the configured batch size. After saving all
models' answers, it grades them using a process pool driving Modal sandboxes.

`run_codex.py` loads the **latest prepared OpenRouter run**, preserving its exact
prompts, secrets, cipher and private tests. It runs `gpt-5.6-luna` through a
process pool, saves every answer, then grades with a Modal process pool.
It does not generate new inputs or call OpenRouter. Run it after the first
command and before preparing another OpenRouter run to compare the same inputs.

Both write under `$STEGO_ARTIFACTS_DIR/tinker/<timestamp>/`. The artifact-relative
path is saved in `tinker/latest_run.txt` for the second script. Files include:

- `config.json`, `queries.jsonl`: shared settings, prompts, secrets and grading cases.
- `openrouter-gpt-oss-120b.jsonl`, `openrouter-gpt-oss-20b.jsonl`, `openrouter-gpt-5.6-luna.jsonl`: OpenRouter raw answers and errors.
- `gpt-5.6-luna.jsonl`: Codex Luna raw answers and errors, separate from OpenRouter Luna.
- `openrouter-results.jsonl`, `codex-results.jsonl`: Modal verdicts and decoding results.
- `openrouter-summary.json`, `codex-summary.json`: functional, secret and joint success counts.
- `openrouter-timings.json`, `codex-timings.json`: key check, preparation/loading, generation, grading and total wall times.
- `codex-config.json`, `codex/`: Codex settings and SDK request/answer artifacts.

OpenRouter answer records include batch wall time repeated for each answer in that
batch; do not sum those repeated values. Individual API latency is unavailable and
stored as null. Luna generation and Modal grading records include per-job wall time.
Both scripts print stage timings and success counts. Errors are saved and remain
in the total denominator. Private tests go only to Modal; code never executes locally.

Progress bars appear automatically: OpenRouter shows completed batches for each
model; Codex generation and Modal grading show completed results, including errors.
Each bar shows elapsed time, completion rate and an estimated time remaining once
completions arrive. OpenRouter updates after whole batches finish, so it can stay at
zero until the first batch returns. No debug logging or extra flags are needed.

During each stage, completed results go to separate UUID-named files in a temporary
directory beside the final JSONL file. Workers never share a result file or lock.
Once the pool stops, a `finally` block concatenates completed files, also on errors
or Ctrl-C, then removes the temporary directory. Incomplete writes are excluded.
The final file stays empty until this merge; record order is arbitrary, so join by
model and problem ID. If merging fails or the process is killed without cleanup,
the temporary directory remains available for recovery.

Settings are hardcoded near the top of each script. There are no retries or resume;
each OpenRouter invocation creates a fresh run, and an existing Luna answer file
prevents rerunning Luna in that directory. Luna's helper does not apply the
OpenRouter token limit. Provider reasoning and
sampling defaults apply. OpenRouter Luna adds a comparison without the Codex scaffold;
it does not equalize all provider settings. The shared cipher uses one control bit and four length
bits, followed by the three-bit payload; ordered synonym pairs emit 0 or 1.
