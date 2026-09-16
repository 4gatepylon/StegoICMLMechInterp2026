"""Generate model-by-model with LiteLLM, then independently grade on Modal."""

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import litellm
from pydantic import BaseModel, ConfigDict, Field

from ciphers.variable_naming_in_python_v2.data.modal_apps import ModalAppsConfig
from ciphers.variable_naming_in_python_v2.tinker.screening_evaluate import (
    CandidateResult,
    ChatResponse,
    ExecutionConfig,
    InputFingerprint,
    SavedResponse,
    _load_cache,
    run_custom_prepared,
    summarize,
)
from ciphers.variable_naming_in_python_v2.tinker.screening_prepare import PreparedRequest, RunConfig, artifact_path
from lib.utils.profiling import Profiler, measure, report_duration, summarize_timings

__all__ = ["generate_prepared", "grade_prepared", "run_prepared", "summarize", "summarize_profile"]


class GenerationConfig(BaseModel):
    """Persist generation.json; batch_size bounds both submissions and SDK threads.

    Models execute sequentially in RunConfig order. Each batch contains up to
    batch_size prompts for one model, with no application retries or fallbacks.
    A resumed generation may change batch_size without changing saved inputs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    batch_size: int = Field(default=100, ge=1, strict=True)


def _inputs(directory: Path) -> tuple[RunConfig, list[PreparedRequest], InputFingerprint]:
    """Read immutable preparation files and validate identity before any calls.

    Returns the saved RunConfig, ordered PreparedRequest rows, and a fingerprint
    of config/requests/grading_cases bytes. Models must belong to the config and
    token limits must match it, since each LiteLLM batch shares those settings.
    Existing fingerprints reject edits; legacy runs rely on unchanged originals.
    """
    config = RunConfig.model_validate_json((directory / "config.json").read_text())
    requests = [PreparedRequest.model_validate_json(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
    if not requests or len({row.request_id for row in requests}) != len(requests):
        raise ValueError("Prepared requests must be nonempty with unique request IDs")
    if any(row.body.model not in config.models or row.body.max_tokens != config.max_tokens for row in requests):
        raise ValueError("Prepared model/token settings differ from config")
    fingerprint = InputFingerprint(
        **{
            key: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
            for key, filename in (("config", "config.json"), ("requests", "requests.jsonl"), ("grading_cases", "grading_cases.json"))
        }
    )
    path = directory / "input_fingerprint.json"
    if path.exists() and InputFingerprint.model_validate_json(path.read_text()) != fingerprint:
        raise ValueError("Prepared inputs changed; resume requires the original files")
    return config, requests, fingerprint


def _archive_error(directory: Path) -> None:
    """Move a prior error.json into append-only errors.jsonl before a new attempt."""
    path = directory / "error.json"
    if path.exists():
        with (directory / "errors.jsonl").open("a") as history:
            history.write(json.dumps(json.loads(path.read_text())) + "\n")
        path.unlink()


def response_json(response: litellm.ModelResponse) -> dict:
    """Serialize a LiteLLM completion without inventing billed cost.

    Returns the SDK response dictionary consumed by ChatResponse: one choices
    entry with message.content/finish_reason and optional usage token counts/cost.
    Extra SDK fields, including reasoning/provider metadata, are retained. If
    LiteLLM moved OpenRouter's billed cost into its provider header, copy that
    reported value into usage.cost for the existing summary; estimates are unused.
    Missing usage becomes an empty object. Raw HTTP bytes/headers are not archived.
    """
    raw = response.model_dump(mode="json")
    if raw.get("usage") is None:
        raw["usage"] = {}
    headers = response._hidden_params.get("additional_headers", {})
    cost = headers.get("llm_provider-x-litellm-response-cost")
    if raw["usage"].get("cost") is None and cost is not None:
        raw["usage"]["cost"] = float(cost)
    return raw


def generate_prepared(run_dir: Path, *, approved: bool = False, batch_size: int = 100, resume: bool = False) -> list[SavedResponse]:
    """Save all answers through LiteLLM/OpenRouter without creating Modal workers.

    ``run_dir`` is relative to STEGO_ARTIFACTS_DIR; ``approved=True`` authorizes
    the reviewed requests. ``batch_size`` bounds prompts/threads per model batch.
    The saved model IDs stay unchanged; only SDK dispatch adds ``openrouter/``.
    Models run sequentially; prompts within each batch run concurrently. Timeout
    comes from RunConfig. No JSON schema, tools, repairs, or retries are added.
    ``resume=True`` reuses validated answers and requests only missing responses;
    False rejects already-started runs. Never run two invocations on one directory.

    Returns SavedResponse rows in prepared order. responses.jsonl contains
    request_id and the full serialized SDK response (see response_json). Successful
    answers in a partially failed batch are flushed before raising its first error;
    subsequent batches/models are not submitted, and no grading occurs. Error
    envelopes are recorded in error.json rather than cached as usable answers.
    generation.json records the initial settings; generations.jsonl records resumes.
    profile.jsonl saves local batch/model/stage and SDK request durations. SDK timings
    exclude thread-queue wait; absent SDK timings remain unknown. A killed process
    can lose a batch's unsaved responses, which may be billed again on resume.
    """
    if approved is not True:
        raise ValueError("Review estimate.json, then explicitly pass approved=True")
    stage_started = perf_counter()
    execution = GenerationConfig(batch_size=batch_size)
    directory = artifact_path(run_dir)
    config, requests, fingerprint = _inputs(directory)
    response_path = directory / "responses.jsonl"
    if resume and not response_path.exists():
        raise FileNotFoundError("No started run to resume")
    if not resume and response_path.exists():
        raise FileExistsError("Run already started; use resume=True")
    cached, _ = _load_cache(directory, requests) if resume else ({}, {})
    missing = [row for row in requests if row.request_id not in cached]
    if not missing:
        return [SavedResponse(request_id=row.request_id, response=cached[row.request_id]) for row in requests]
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("Set OPENROUTER_API_KEY before inference")
    preflight_s = perf_counter() - stage_started
    profiler = Profiler(directory / "profile.jsonl")
    (directory / "input_fingerprint.json").write_text(fingerprint.model_dump_json(indent=2))
    if resume:
        _archive_error(directory)
        with (directory / "generations.jsonl").open("a") as history:
            history.write(execution.model_dump_json() + "\n")
    else:
        (directory / "generation.json").write_text(execution.model_dump_json(indent=2))
    current_id = missing[0].request_id
    with profiler.bind(), measure("generation.total", items=len(missing), start_s=stage_started):
        report_duration("generation.preflight", preflight_s, source="local")
        try:
            with response_path.open("a" if resume else "x") as output:
                for model in config.models:
                    pending = [row for row in missing if row.body.model == model]
                    if not pending:
                        continue
                    with profiler.bind(model=model), measure("generation.model", items=len(pending)):
                        for start in range(0, len(pending), execution.batch_size):
                            batch = pending[start : start + execution.batch_size]
                            current_id = batch[0].request_id
                            with measure("generation.batch", items=len(batch)):
                                with measure("generation.dispatch"):
                                    responses = litellm.batch_completion(
                                        model=f"openrouter/{model}",
                                        messages=[[message.model_dump() for message in row.body.messages] for row in batch],
                                        max_tokens=config.max_tokens,
                                        timeout=config.timeout_s,
                                        max_workers=min(execution.batch_size, len(batch)),
                                        num_retries=0,
                                    )
                                if len(responses) != len(batch):
                                    raise RuntimeError("LiteLLM returned a different number of responses than requests")
                                failures = []
                                for request, response in zip(batch, responses, strict=True):
                                    with profiler.bind(model=model, request_id=request.request_id):
                                        try:
                                            with measure("generation.validate_response"):
                                                if isinstance(response, Exception):
                                                    raise response
                                                raw = response_json(response)
                                                parsed = ChatResponse.model_validate(raw)
                                                if parsed.error is not None or len(parsed.choices) != 1:
                                                    raise RuntimeError("Invalid API completion: expected one choice without API error")
                                            if response._response_ms is not None:
                                                report_duration("generation.request", response._response_ms / 1000, source="sdk", items=1)
                                            for key in ("litellm_overhead_time_ms", "callback_duration_ms"):
                                                if (duration := response._hidden_params.get(key)) is not None:
                                                    report_duration(f"generation.{key.removesuffix('_ms')}", duration / 1000, source="sdk")
                                            with measure("generation.persist", items=1):
                                                output.write(SavedResponse(request_id=request.request_id, response=raw).model_dump_json() + "\n")
                                                output.flush()
                                            cached[request.request_id] = raw
                                        except Exception as error:
                                            failures.append((request.request_id, error))
                                if failures:
                                    current_id, error = failures[0]
                                    raise error
        except BaseException as error:
            (directory / "error.json").write_text(json.dumps({"request_id": current_id, "stage": "generation", "error": f"{type(error).__name__}: {error}"}, indent=2))
            raise
    return [SavedResponse(request_id=row.request_id, response=cached[row.request_id]) for row in requests]


def grade_prepared(run_dir: Path, *, approved: bool = False, num_workers: int = 16, resume: bool = False, modal_config: ModalAppsConfig | None = None) -> list[CandidateResult]:
    """Grade a complete response cache on Modal without any generation calls.

    ``run_dir`` and ``approved`` have the generation-stage contract. ``num_workers``
    bounds independent candidate graders; ``modal_config`` controls each sandbox.
    ``resume=True`` keeps completed verdicts and grades only unfinished candidates;
    False rejects an already-started grading stage. Every prepared answer must be
    saved before grading begins. Modal settings/inputs cannot change on resume.
    Returns CandidateResult rows in prepared order and persists results.jsonl in
    completion order. Uses the existing custom runner's cache/error rules with a
    generator that always raises. Profiles include candidate and Modal substeps;
    parsing failures count as completed candidate failures, not infrastructure errors.
    """
    if approved is not True:
        raise ValueError("Explicitly pass approved=True for Modal grading")
    stage_started = perf_counter()
    execution = ExecutionConfig(num_workers=num_workers, modal_config=modal_config or ModalAppsConfig())
    directory = artifact_path(run_dir)
    _, requests, _ = _inputs(directory)
    cached, results = _load_cache(directory, requests)
    if len(cached) != len(requests):
        raise ValueError("Generate and save all responses before grading")
    if not resume and (directory / "results.jsonl").exists():
        raise FileExistsError("Grading already started; use resume=True")
    settings_path = directory / "execution.json"
    if settings_path.exists() and ExecutionConfig.model_validate_json(settings_path.read_text()).modal_config != execution.modal_config:
        raise ValueError("Modal settings must remain unchanged when resuming")
    if len(results) == len(requests):
        return [results[index] for index in range(len(requests))]
    preflight_s = perf_counter() - stage_started
    profiler = Profiler(directory / "profile.jsonl")
    if not settings_path.exists():
        settings_path.write_text(execution.model_dump_json(indent=2))

    def no_generation(request: PreparedRequest, timeout_s: int) -> dict:
        """Reject a missing cached answer; grading may never generate a replacement."""
        raise RuntimeError(f"Missing saved response for {request.request_id}")

    with profiler.bind(), measure("grading.total", items=len(requests) - len(results), start_s=stage_started):
        report_duration("grading.preflight", preflight_s, source="local")
        return run_custom_prepared(run_dir, approved=True, num_workers=num_workers, resume=True, modal_config=execution.modal_config, generate=no_generation, profiler=profiler)


def run_prepared(
    run_dir: Path, *, approved: bool = False, batch_size: int = 100, num_workers: int = 16, resume: bool = False, modal_config: ModalAppsConfig | None = None
) -> list[CandidateResult]:
    """Generate every model sequentially, then grade saved answers concurrently.

    Arguments match generate_prepared and grade_prepared. Validate both stage
    settings before paid work. Resume uses existing answers/grades; a completed run
    makes no calls or new profile rows. Return ordered CandidateResult records.
    """
    execution = ExecutionConfig(num_workers=num_workers, modal_config=modal_config or ModalAppsConfig())
    directory = artifact_path(run_dir)
    settings = directory / "execution.json"
    if resume and settings.exists() and ExecutionConfig.model_validate_json(settings.read_text()).modal_config != execution.modal_config:
        raise ValueError("Modal settings must remain unchanged when resuming")
    generate_prepared(run_dir, approved=approved, batch_size=batch_size, resume=resume)
    return grade_prepared(run_dir, approved=approved, num_workers=num_workers, resume=resume, modal_config=execution.modal_config)


def summarize_profile(run_dir: Path) -> list[dict]:
    """Return notebook-ready timing aggregates from this run's profile.jsonl.

    ``run_dir`` is relative to STEGO_ARTIFACTS_DIR. See summarize_timings for all
    returned keys and inclusive-duration/throughput semantics. Missing legacy
    profiles return []; each resume is grouped separately by invocation_id.
    No requests, grading, or writes occur.
    """
    return summarize_timings(artifact_path(run_dir) / "profile.jsonl")
