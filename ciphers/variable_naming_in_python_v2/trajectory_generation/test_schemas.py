"""Contract tests for interface-only trajectory records; no service/notebook tests.

Partitions: payload endpoints 1/8 and invalid lengths; probability endpoints 0/1,
outside-range and nonfinite inputs; ordinary versus framed control 0/1; completed
valid/malformed outputs versus infrastructure errors; valid versus contradictory
cross-field records. JSON round trips exercise consumer serialization, including
nullable targets, leading zeroes, source whitespace, and artifact-relative paths.
Runtime sampling, prompts, decoder correctness, concurrency, and resume are omitted.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from ciphers.variable_naming_in_python_v2.data.apps import AppsTestCases
from ciphers.variable_naming_in_python_v2.data.codex_apps import AppsPromptProblem
from ciphers.variable_naming_in_python_v2.decoder import DecodedMessage
from ciphers.variable_naming_in_python_v2.trajectory_generation.__main__ import main
from ciphers.variable_naming_in_python_v2.trajectory_generation.schemas import (
    GenerationRecord,
    GradeRecord,
    ProblemRecord,
    RequestRecord,
    RunConfig,
    RunManifest,
    SamplingConfig,
    StageSummary,
)

EXAMPLE_CONFIG = Path("ciphers/variable_naming_in_python_v2/trajectory_generation/config.example.json")
NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)


def request_record(condition: str = "framed", control: int = 1, bits: str = "001") -> RequestRecord:
    """Build a consumer record for one condition; control/bits apply only to framed.

    condition selects ordinary or framed. Return a validated RequestRecord with
    matching nullable indices/targets; tests mutate serialized copies to exercise
    rejection without bypassing Pydantic validation through model_copy.
    """
    ordinary = condition == "ordinary"
    return RequestRecord(
        request_id="p7-o0" if ordinary else "p7-l0-b0",
        problem_id=7,
        condition=condition,
        ordinary_sample_index=0 if ordinary else None,
        length_sample_index=None if ordinary else 0,
        bitstring_sample_index=None if ordinary else 0,
        sampled_length=None if ordinary else len(bits),
        message_bits=None if ordinary else bits,
        control_bit=None if ordinary else control,
        prompt="Solve the supplied public task.",
    )


def grade_record(condition: str = "framed", *, absent: bool = False) -> GradeRecord:
    """Build a completed grade for testing the nullable outcome contract.

    condition controls whether encoding/joint targets exist. absent chooses a
    valid absent or present DecodedMessage. Return GradeRecord with no Modal verdict
    because these tests check serialization/field consistency, not actual grading.
    """
    return GradeRecord(
        request_id="p7-o0" if condition == "ordinary" else "p7-l0-b0",
        condition=condition,
        status="completed",
        code_sha256="a" * 64,
        functional_success=True,
        encoding_success=True if condition == "framed" else None,
        joint_success=True if condition == "framed" else None,
        verdict=None,
        decoded=DecodedMessage(is_encoding=not absent, length=None if absent else 3, message_bits=None if absent else "001", bindings=()),
        decode_error=None,
        error=None,
        elapsed_seconds=1.5,
    )


def generation_record() -> GenerationRecord:
    """Return a completed source record preserving Unicode, spaces, and final newline."""
    return GenerationRecord(
        request_id="p7-l0-b0",
        status="completed",
        text='{"code":"# café\\nx = 1  \\n"}',
        code="# café\nx = 1  \n",
        output_error=None,
        error=None,
        resolved_model="configured-model",
        turn_id="turn-example",
        sdk_artifact_subdir=Path("datasets/apps/trajectory-generation/demo/sdk/example"),
        started_at=NOW,
        finished_at=NOW,
        elapsed_seconds=0.0,
    )


def test_master_config_and_manifest_round_trip():
    """All-tier filters/model overrides survive the manifest consumed by later stages."""
    values = RunConfig.model_validate_json(EXAMPLE_CONFIG.read_text()).model_dump(mode="json")
    values["apps"].update(split="test", min_lines=2, max_lines=100, min_chars=10, max_chars=5000, min_tests=3)
    values["inference"]["model"] = "another-model"
    values["sampling"].update(payload_one_probability=0.2, control_one_probability=0.8, ordinary_generations_per_problem=3)
    config = RunConfig.model_validate(values)
    per_problem = config.sampling.lengths_per_problem * config.sampling.bitstrings_per_length + 3
    manifest = RunManifest(run_id="demo", created_at=NOW, repository_commit=None, config=config, problem_count=2, request_count=2 * per_problem)
    restored = RunManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest
    assert set(restored.config.apps.difficulties) == {"introductory", "interview", "competition"}
    invalid = manifest.model_dump()
    invalid["request_count"] -= 1
    with pytest.raises(ValidationError, match="sampling budget"):
        RunManifest.model_validate(invalid)


@pytest.mark.parametrize("probability", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("length", [1, 8])
def test_sampling_valid_partitions(probability, length):
    config = SamplingConfig(min_bits=length, max_bits=length, payload_one_probability=probability, control_one_probability=1 - probability)
    assert SamplingConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize(
    "override",
    [
        {"min_bits": 0},
        {"max_bits": 9},
        {"min_bits": 8, "max_bits": 1},
        {"lengths_per_problem": 0},
        {"bitstrings_per_length": 0},
        {"ordinary_generations_per_problem": -1},
        {"payload_one_probability": -0.1},
        {"control_one_probability": 1.1},
        {"payload_one_probability": float("nan")},
        {"control_one_probability": float("inf")},
        {"min_bits": True},
        {"seed": "42"},
        {"unknown_setting": 1},
    ],
)
def test_sampling_rejects_invalid_domains(override):
    with pytest.raises(ValidationError):
        SamplingConfig.model_validate(override)


def test_cipher_capacity_and_baseline_disable():
    values = RunConfig.model_validate_json(EXAMPLE_CONFIG.read_text()).model_dump(mode="json")
    values["sampling"]["ordinary_generations_per_problem"] = 0
    values["cipher"]["length_bits"] = 3
    with pytest.raises(ValidationError, match="cannot represent"):
        RunConfig.model_validate(values)
    values["sampling"]["max_bits"] = 7
    config = RunConfig.model_validate(values)
    assert RunConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize("condition,control,bits", [("ordinary", 0, "0"), ("framed", 0, "00000000"), ("framed", 1, "0"), ("framed", 1, "00100001")])
def test_request_round_trip(condition, control, bits):
    record = request_record(condition, control, bits)
    assert RequestRecord.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize(
    "override",
    [{"sampled_length": 2}, {"message_bits": "01x"}, {"control_bit": True}, {"control_bit": 2}, {"ordinary_sample_index": 0}, {"prompt": "   "}, {"request_id": "../escape"}],
)
def test_request_rejects_inconsistent_or_unsafe_fields(override):
    values = request_record().model_dump()
    values.update(override)
    with pytest.raises(ValidationError):
        RequestRecord.model_validate(values)


@pytest.mark.parametrize("field,value", [("message_bits", "001"), ("control_bit", 0), ("ordinary_sample_index", None)])
def test_ordinary_requests_cannot_have_framed_targets(field, value):
    values = request_record("ordinary").model_dump()
    values[field] = value
    with pytest.raises(ValidationError):
        RequestRecord.model_validate(values)


def test_problem_round_trip_and_invocation_contract():
    problem = ProblemRecord(
        problem_id=7,
        difficulty="competition",
        public_problem=AppsPromptProblem(question="Double x.", fn_name="solve"),
        test_cases=AppsTestCases(inputs=[[2]], outputs=[4], fn_name="solve"),
    )
    assert ProblemRecord.model_validate_json(problem.model_dump_json()) == problem
    values = problem.model_dump()
    values["public_problem"]["fn_name"] = None
    with pytest.raises(ValidationError, match="same fn_name"):
        ProblemRecord.model_validate(values)
    values["public_problem"]["fn_name"] = "solve"
    values["test_cases"].update(inputs=[], outputs=[])
    with pytest.raises(ValidationError, match="at least one"):
        ProblemRecord.model_validate(values)


def test_generation_round_trip_malformed_and_infrastructure_partitions():
    record = generation_record()
    assert GenerationRecord.model_validate_json(record.model_dump_json()) == record
    values = record.model_dump()
    values.update(code="x =", output_error="SyntaxError")
    assert GenerationRecord.model_validate(values).code == "x ="
    values.update(status="error", error="Transport failed", output_error=None)
    error = GenerationRecord.model_validate(values)
    assert GenerationRecord.model_validate_json(error.model_dump_json()) == error
    values["error"] = None
    with pytest.raises(ValidationError):
        GenerationRecord.model_validate(values)


@pytest.mark.parametrize("path", ["/tmp/example", "datasets/../escape"])
def test_generation_paths_stay_artifact_relative(path):
    values = generation_record().model_dump()
    values["sdk_artifact_subdir"] = path
    with pytest.raises(ValidationError):
        GenerationRecord.model_validate(values)


@pytest.mark.parametrize("condition", ["ordinary", "framed"])
@pytest.mark.parametrize("absent", [True, False])
def test_grade_round_trip_and_error_nullability(condition, absent):
    record = grade_record(condition, absent=absent)
    assert GradeRecord.model_validate_json(record.model_dump_json()) == record
    values = record.model_dump()
    values.update(status="error", error="Modal connection failed", functional_success=None, encoding_success=None, joint_success=None)
    error = GradeRecord.model_validate(values)
    assert GradeRecord.model_validate_json(error.model_dump_json()) == error
    values["functional_success"] = False
    with pytest.raises(ValidationError, match="null success"):
        GradeRecord.model_validate(values)


@pytest.mark.parametrize(
    "condition,override",
    [
        ("ordinary", {"encoding_success": True}),
        ("ordinary", {"joint_success": False}),
        ("framed", {"encoding_success": None}),
        ("framed", {"joint_success": False}),
        ("framed", {"functional_success": 1}),
        ("framed", {"decode_error": "error despite decoded result"}),
    ],
)
def test_grade_rejects_ambiguous_success(condition, override):
    values = grade_record(condition).model_dump()
    values.update(override)
    with pytest.raises(ValidationError):
        GradeRecord.model_validate(values)


def test_stage_summary_cannot_hide_pending_work():
    summary = StageSummary(stage="generate", eligible_count=5, completed_count=2, skipped_count=2, error_count=1, elapsed_seconds=3.0)
    assert StageSummary.model_validate_json(summary.model_dump_json()) == summary
    values = summary.model_dump()
    values["eligible_count"] = 6
    with pytest.raises(ValidationError):
        StageSummary.model_validate(values)


def test_cli_help_and_explicit_unimplemented_failure():
    runner = CliRunner()
    help_result = runner.invoke(main, ["--help"])
    assert help_result.exit_code == 0
    assert "--ordinary-generations-per-problem" in help_result.output
    result = runner.invoke(main, ["--run-id", "demo", "--steps", "generate"])
    assert result.exit_code != 0
    assert "not implemented" in result.output
