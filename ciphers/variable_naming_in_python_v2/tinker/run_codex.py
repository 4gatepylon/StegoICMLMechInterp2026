"""Run after run_openrouter: python -m ciphers.variable_naming_in_python_v2.tinker.run_codex.

Requires STEGO_ARTIFACTS_DIR, Codex ChatGPT login and Modal credentials. Reads the
latest prepared OpenRouter inputs and writes Luna outputs into the same directory.
No API key, retries or resume. Codex reasoning defaults apply; the OpenRouter
token limit is not applied by the existing Codex helper. Code executes only on Modal.

NOTE: a human has only SKIMMED this.
"""

import asyncio
import json
import os
from functools import partial
from pathlib import Path
from time import perf_counter

from ciphers.variable_naming_in_python_v2.data.apps import REPO_ROOT
from ciphers.variable_naming_in_python_v2.data.codex_apps import CodexInferenceConfig, infer
from ciphers.variable_naming_in_python_v2.decoder import CipherConfig
from ciphers.variable_naming_in_python_v2.tinker.run_openrouter import Query, finish, grade, stage

MODEL, WORKERS, MODAL_WORKERS = "gpt-5.6-luna", 32, 16


def generate(query: Query, *, config: CodexInferenceConfig) -> dict:
    """Generate one Luna answer from query.prompt with config's limits/artifact path.

    Returns problem_id/model join keys, text (unparsed answer), raw (SDK response
    or None), error (exception or None), and seconds (wall time). The parent saves
    all records before grading; failures are recorded without retrying.
    """
    start = perf_counter()
    result = dict(problem_id=query.problem_id, model=config.model, text="", raw=None, error=None)
    try:
        response = asyncio.run(infer(query.prompt, config, response_format="text"))
        result.update(text=response.text, raw=response.model_dump(mode="json"))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return dict(result, seconds=perf_counter() - start)


def main() -> None:
    """Load the latest prepared inputs, generate Luna answers, and grade on Modal.

    tinker/latest_run.txt supplies the artifact-relative run directory. Its
    config.json requires cipher (CipherConfig); queries.jsonl contains Query rows.
    These saved inputs preserve exact prompts, secrets and private tests even if
    current preparation settings change. Outputs have codex/luna names, so the
    OpenRouter files remain available alongside them. An existing Luna file fails.
    """
    start, timings = perf_counter(), {}
    artifacts = REPO_ROOT / Path(os.environ["STEGO_ARTIFACTS_DIR"])
    run_dir = Path((artifacts / "tinker/latest_run.txt").read_text().strip())
    config = CodexInferenceConfig(model=MODEL, timeout_s=600, artifact_subdir=run_dir / "codex")
    directory = artifacts / run_dir
    print(f"Using saved inputs in {directory}", flush=True)
    queries = [Query.model_validate_json(line) for line in (directory / "queries.jsonl").read_text().splitlines()]
    cipher = CipherConfig.model_validate(json.loads((directory / "config.json").read_text())["cipher"])
    (directory / "codex-config.json").write_text(config.model_dump_json(indent=2))
    timings["load_inputs"] = perf_counter() - start
    print(f"load_inputs: {len(queries)} queries in {timings['load_inputs']:.1f}s", flush=True)
    answers = stage(partial(generate, config=config), queries, WORKERS, directory / "gpt-5.6-luna.jsonl", timings)
    results = stage(partial(grade, queries={q.problem_id: q for q in queries}, cipher=cipher), answers, MODAL_WORKERS, directory / "codex-results.jsonl", timings)
    finish(results, directory, "codex", timings, start)


if __name__ == "__main__":
    main()
