"""Exercise APPS selection and loading without running any supplied program.

Partitions cover size bounds below/at/above cutoffs; one/many/no acceptable
references; selected/unselected difficulty tiers; stdio/structured function
cases; absent/malformed/unequally paired tests; and empty/nonempty results.
The source adapter is checked with local Parquet and mocked Hub downloads.
These tests omit remote availability, reference correctness, test execution,
semantic test coverage, and notebooks.
"""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from datasets import Dataset
from pydantic import ValidationError

from ciphers.variable_naming_in_python_v2.data import apps
from ciphers.variable_naming_in_python_v2.data.apps import AppsConfig, filter_apps, load_apps


def source_row(solutions: list[str] | None = None, count: int = 10, **changes: object) -> dict[str, object]:
    """Return an official-schema fixture; overrides create selected invalid partitions.

    ``solutions=None`` supplies a 20-line reference string. ``count`` controls
    the number of paired stdio cases. ``changes`` replaces source fields to
    model upstream records. The returned mapping is consumed by ``filter_apps``
    according to its source schema; no fixture text is executed.
    """
    row = {
        "problem_id": 1,
        "question": "Read numbers and print their sum.",
        "solutions": json.dumps(solutions if solutions is not None else ["total = 0\n" * 20]),
        "input_output": json.dumps({"inputs": [str(i) for i in range(count)], "outputs": [str(i) for i in range(count)]}),
        "difficulty": "introductory",
        "url": "https://example.com/problem/1",
        "starter_code": "",
    }
    row.update(changes)
    return row


def test_defaults_select_ground_truth_not_question_or_starter() -> None:
    long_code = "total = 0\n" * 20
    rows = [
        source_row(["pass", long_code]),
        source_row(["pass"], problem_id=2, question=long_code, starter_code=long_code),
        source_row([long_code], problem_id=3, difficulty="interview"),
        source_row([long_code], problem_id=4, count=9),
    ]
    result = filter_apps(rows)
    assert list(result["problem_id"]) == [1]
    assert result[0]["solutions"] == [long_code]
    assert result[0]["num_tests"] == 10


@pytest.mark.parametrize("line_count,kept", [(1, False), (2, True), (3, True), (4, False)])
def test_inclusive_line_bounds(line_count: int, kept: bool) -> None:
    result = filter_apps([source_row(["pass\n" * line_count])], AppsConfig(min_lines=2, max_lines=3))
    assert bool(len(result)) is kept


@pytest.mark.parametrize("chars,kept", [(2, False), (3, True), (4, True), (5, False)])
def test_inclusive_character_bounds(chars: int, kept: bool) -> None:
    result = filter_apps([source_row(["x" * chars])], AppsConfig(min_lines=0, min_chars=3, max_chars=4))
    assert bool(len(result)) is kept


def test_line_and_character_bounds_apply_to_same_answer() -> None:
    rows = [source_row(["a\nb\nc", "long_single_line"])]
    assert len(filter_apps(rows, AppsConfig(min_lines=3, min_chars=10))) == 0


def test_counts_preserve_blank_lines_comments_unicode_and_newlines() -> None:
    code = "# é\r\n\r\nx = 1\r\n"
    result = filter_apps([source_row([code])], AppsConfig(min_lines=3, max_lines=3, min_chars=len(code), max_chars=len(code)))
    assert result[0]["solutions"] == [code]


def test_disabled_upper_bounds_allow_large_answers() -> None:
    code = "value = 123456789\n" * 1000
    result = filter_apps([source_row([code])], AppsConfig(min_lines=1, min_chars=1))
    assert result[0]["solutions"] == [code]


@pytest.mark.parametrize("solutions", [[], [""], [" \n\t"], ["pass"]])
def test_missing_or_nonqualifying_ground_truth_is_excluded(solutions: list[str]) -> None:
    assert len(filter_apps([source_row(solutions)])) == 0


def test_blank_answers_still_excluded_when_all_minimums_disabled() -> None:
    assert len(filter_apps([source_row([" \n"], count=0)], AppsConfig(min_lines=0, min_tests=0))) == 0


def test_multiple_difficulty_selection_and_reference_order() -> None:
    rows = [source_row(["first", "second"], problem_id=i, difficulty=level) for i, level in enumerate(["introductory", "interview", "competition"])]
    config = AppsConfig(difficulties=("introductory", "competition"), min_lines=0)
    result = filter_apps(rows, config)
    assert list(result["problem_id"]) == [0, 2]
    assert result[1]["solutions"] == ["first", "second"]


@pytest.mark.parametrize("count,kept", [(2, False), (3, True), (4, True)])
def test_configurable_test_pair_threshold(count: int, kept: bool) -> None:
    assert bool(len(filter_apps([source_row(count=count)], AppsConfig(min_tests=3)))) is kept


def test_heterogeneous_json_cases_survive_arrow_storage() -> None:
    cases = {"inputs": [[1, {"nested": [True, None, "é"]}], ["text"]], "outputs": [{"ok": True}, 5], "fn_name": "solve"}
    result = filter_apps([source_row(count=2), source_row(problem_id=2, input_output=json.dumps(cases))], AppsConfig(min_tests=2))
    assert result[0]["input_output"] == {"inputs": ["0", "1"], "outputs": ["0", "1"], "fn_name": None}
    assert result[1]["input_output"] == cases
    assert result[1]["num_tests"] == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"solutions": "not json"},
        {"solutions": "null"},
        {"solutions": "[123]"},
        {"input_output": ""},
        {"input_output": "null"},
        {"input_output": "{}"},
        {"input_output": '{"inputs": [1], "outputs": []}'},
        {"input_output": '{"inputs": "x", "outputs": "y"}'},
        {"question": "   "},
    ],
)
def test_malformed_rows_do_not_count_as_tested_problems(changes: dict[str, object]) -> None:
    result = filter_apps([source_row(**changes)], AppsConfig(min_lines=0, min_tests=0))
    assert len(result) == 0
    assert "solutions" in result.column_names


def test_missing_solutions_and_empty_input_keep_output_schema(caplog: pytest.LogCaptureFixture) -> None:
    row = source_row()
    del row["solutions"]
    rejected = filter_apps([row])
    empty = filter_apps([])
    assert rejected.features == empty.features == filter_apps([source_row()]).features
    assert "Skipped 1 malformed APPS rows" in caplog.text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_lines": -1},
        {"min_chars": -1},
        {"min_tests": -1},
        {"max_lines": 19},
        {"min_chars": 5, "max_chars": 4},
        {"max_chars": -1},
        {"min_lines": 1.5},
        {"min_tests": True},
        {"difficulties": ("easy",)},
        {"difficulties": ()},
        {"cache_dir": "../outside"},
        {"cache_dir": "/outside"},
        {"unknown_option": 1},
    ],
)
def test_invalid_configuration_fails_early(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AppsConfig(**kwargs)


def test_load_parquet_uses_selected_split_and_artifact_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parquet = tmp_path / "source.parquet"
    Dataset.from_list([source_row()]).to_parquet(parquet)
    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    repository = Mock()
    repository.list_repo_files.return_value = ["introductory/train/0000.parquet", "introductory/test/0000.parquet", "interview/test/0000.parquet"]
    monkeypatch.setattr(apps, "HfApi", lambda: repository)
    download = Mock(return_value=str(parquet))
    monkeypatch.setattr(apps, "hf_hub_download", download)
    result = load_apps(AppsConfig(split="test", difficulties=("introductory", "introductory"), revision="snapshot"))
    assert len(result) == 1
    download.assert_called_once_with(
        "codeparrot/apps",
        "introductory/test/0000.parquet",
        repo_type="dataset",
        revision="snapshot",
        cache_dir=str(tmp_path / "datasets/apps/hub"),
    )
    assert (tmp_path / "datasets/apps/arrow").is_dir()


def test_load_requires_artifact_root_before_contacting_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STEGO_ARTIFACTS_DIR", raising=False)
    hub = Mock()
    monkeypatch.setattr(apps, "HfApi", hub)
    with pytest.raises(ValueError, match="STEGO_ARTIFACTS_DIR"):
        load_apps()
    hub.assert_not_called()


def test_load_rejects_missing_source_partition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    repository = Mock()
    repository.list_repo_files.return_value = []
    monkeypatch.setattr(apps, "HfApi", lambda: repository)
    with pytest.raises(ValueError, match="No APPS Parquet files"):
        load_apps()
