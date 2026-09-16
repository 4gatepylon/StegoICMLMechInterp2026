"""Run Luna on the exact saved screening inputs with the shared threaded grader."""

import asyncio
import json
from pathlib import Path

from ciphers.variable_naming_in_python_v2.data.codex_apps import CodexInferenceConfig, infer
from ciphers.variable_naming_in_python_v2.data.modal_apps import ModalAppsConfig
from ciphers.variable_naming_in_python_v2.tinker.openrouter_evaluate import ExecutionConfig, run_prepared
from ciphers.variable_naming_in_python_v2.tinker.openrouter_prepare import PreparedRequest, RunConfig, artifact_path

LUNA_MODEL = "gpt-5.6-luna"


def prepare_codex_comparison(run_dir: Path) -> Path:
    """Copy one saved prompt per problem into a sibling-model comparison run.

    ``run_dir`` is a prepared OpenRouter run relative to STEGO_ARTIFACTS_DIR.
    Returns ``run_dir / 'codex-luna'``, also relative to that root. Creates:
    config.json (same RunConfig with models set to Luna), requests.jsonl (same
    PreparedRequest bodies with only model/request_id changed), grading_cases.json
    (byte-identical private cases), comparison.json (source_run_dir and documented
    provider differences), and estimate.json (model/available/requests fields used
    by summarize, with nullable USD estimates because Codex uses a subscription).

    Repeated model rows collapse by problem ID, preserving first-appearance order.
    All copies of a problem must have identical messages and max_tokens. Every
    problem must have grading cases, and the distinct count must match num_problems.
    Reject an existing comparison directory rather than overwriting any run.
    No inference or grading occurs. The saved max_tokens is retained for provenance;
    the Codex SDK does not expose an equivalent output-token limit here.
    """
    source = artifact_path(run_dir)
    config = RunConfig.model_validate_json((source / "config.json").read_text())
    unique: dict[int, PreparedRequest] = {}
    for line in (source / "requests.jsonl").read_text().splitlines():
        request = PreparedRequest.model_validate_json(line)
        previous = unique.get(request.problem_id)
        if previous is not None:
            if previous.body.messages != request.body.messages or previous.body.max_tokens != request.body.max_tokens:
                raise ValueError(f"Different inputs for problem {request.problem_id}")
        else:
            unique[request.problem_id] = request
    cases_text = (source / "grading_cases.json").read_text()
    cases = json.loads(cases_text)
    if len(unique) != config.num_problems or set(cases) != {str(problem_id) for problem_id in unique}:
        raise ValueError("Distinct saved problems must match num_problems and grading cases")
    target = run_dir / "codex-luna"
    directory = artifact_path(target)
    directory.mkdir(parents=True, exist_ok=False)
    comparison_config = config.model_copy(update={"models": (LUNA_MODEL,)})
    (directory / "config.json").write_text(comparison_config.model_dump_json(indent=2))
    requests = [
        request.model_copy(update={"request_id": f"{request.problem_id}:{LUNA_MODEL}", "body": request.body.model_copy(update={"model": LUNA_MODEL})})
        for request in unique.values()
    ]
    (directory / "requests.jsonl").write_text("".join(request.model_dump_json() + "\n" for request in requests))
    (directory / "grading_cases.json").write_text(cases_text)
    (directory / "estimate.json").write_text(
        json.dumps(
            {
                "models": [{"model": LUNA_MODEL, "available": True, "requests": len(requests)}],
                "estimated_usd": None,
                "limit_scenario_usd": None,
                "billing": "ChatGPT-backed Codex subscription; no OpenRouter charge or per-request USD estimate",
            },
            indent=2,
        )
    )
    (directory / "comparison.json").write_text(
        json.dumps(
            {
                "source_run_dir": str(run_dir),
                "matched": ["problem_ids", "user_prompts", "cipher", "message_bits", "private_tests", "one_answer_per_problem", "response_parser", "decoder"],
                "provider_differences": [
                    "Codex uses the SDK's fixed no-tools base instructions and subscription authentication",
                    "Codex has no matching max_tokens control in this helper; the saved limit is not applied",
                    "Reasoning and sampling use each provider/model's defaults",
                ],
            },
            indent=2,
        )
    )
    return target


def run_codex_comparison(run_dir: Path, *, approved: bool = False, num_workers: int = 16, modal_config: ModalAppsConfig | None = None) -> Path:
    """Run the prepared Luna comparison through the same threaded parser/grader.

    ``run_dir`` is the source OpenRouter run, relative to STEGO_ARTIFACTS_DIR.
    ``approved`` authorizes Codex subscription use and Modal grading. ``num_workers``
    controls concurrency; ``modal_config`` uses the exact same evaluator contract as
    OpenRouter and should match its execution settings. Returns the relative
    comparison directory for summarize(). Inputs must already exist: call
    prepare_codex_comparison(run_dir) first to review them independently of execution.

    Responses use the ChatResponse envelope plus a ``codex`` field containing the
    existing InferenceResult (raw text, exact prompt, turn ID, artifacts, restrictions,
    and preflight snapshot). No USD usage is invented. SDK generation artifacts live
    under the comparison directory's generation/ subdirectory. The configuration is
    also saved as codex_config.json. Each thread runs its own async SDK invocation;
    text mode leaves JSON formatting prompt-driven, exactly as for OpenRouter.
    Existing shared approval, first-error, no-retry, and no-rerun rules apply.
    """
    if approved is not True:
        raise ValueError("Explicitly pass approved=True to run the Codex comparison")
    ExecutionConfig(num_workers=num_workers, modal_config=modal_config or ModalAppsConfig())
    target = run_dir / "codex-luna"
    directory = artifact_path(target)
    config = RunConfig.model_validate_json((directory / "config.json").read_text())
    codex_config = CodexInferenceConfig(model=LUNA_MODEL, timeout_s=config.timeout_s, artifact_subdir=target / "generation")
    (directory / "codex_config.json").write_text(codex_config.model_dump_json(indent=2))

    def generate(request: PreparedRequest, timeout_s: int) -> dict:
        """Adapt one unchanged user prompt to the shared ChatResponse contract.

        ``request`` must name Luna; ``timeout_s`` must equal the saved SDK deadline.
        Returns choices containing the final text and a stop reason (infer only
        returns completed turns), plus the full Codex inference record. SDK errors
        propagate to the shared runner; output parsing is exclusively its job.
        """
        if request.body.model != LUNA_MODEL or timeout_s != codex_config.timeout_s:
            raise ValueError("Saved request differs from the Codex comparison configuration")
        answer = asyncio.run(infer(request.body.messages[0].content, codex_config, response_format="text"))
        return {
            "choices": [{"message": {"content": answer.text}, "finish_reason": "stop"}],
            "codex": answer.model_dump(mode="json"),
        }

    run_prepared(target, approved=True, num_workers=num_workers, modal_config=modal_config, generate=generate)
    return target
