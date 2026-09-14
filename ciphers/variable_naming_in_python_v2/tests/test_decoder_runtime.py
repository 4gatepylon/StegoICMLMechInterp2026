"""Runtime contracts added with extraction and verbose CLI implementation.

Partitions extend the ten fixtures with Unicode/newline spans, additional binding
forms, malformed/unsupported input, and command-line diagnostics. Source is never
executed. Runtime dataflow, dynamic namespaces, and PEP 695 annotation scopes are
not implemented or claimed by these tests.
"""

import logging
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from ciphers.variable_naming_in_python_v2.cli import main
from ciphers.variable_naming_in_python_v2.decoder import BindingOccurrence, CipherConfig, DecodedMessage, SourceSpan, UnsupportedSyntaxError, decode

ONE_GROUP = Path("ciphers/variable_naming_in_python_v2/tests/fixtures/codex_generated_1_group")
ONE_GROUP_CIPHER = CipherConfig.model_validate_json((ONE_GROUP / "cipher.json").read_text(encoding="utf-8"))


def absent(code: str) -> DecodedMessage:
    """Prefix zero control, then decode the supplied code to inspect all bindings.

    Return the complete result under the JSON fixture's i/j cipher. The prefix deliberately
    prevents capacity concerns from hiding binding-resolution regressions; these
    cases assert scope/occurrence facts independently of framing tests.
    """
    return decode("i = 0\n" + code, ONE_GROUP_CIPHER)


def test_identifier_occurrence_rejects_multiline_general_span() -> None:
    span = SourceSpan(line=1, column=4, end_line=3, end_column=2)
    with pytest.raises(ValidationError, match="one physical line"):
        BindingOccurrence(kind="read", span=span)


def test_multiline_expression_still_has_single_line_identifier_occurrences() -> None:
    result = absent("ordinary = (\n    i\n)\n")
    assert [(occ.span.line, occ.span.end_line) for occ in result.bindings[0].occurrences] == [(1, 1), (3, 3)]


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_physical_newlines_and_unicode_string_separators_preserve_positions(newline: str) -> None:
    code = newline.join(['text = "\u2028\f"', "i = 0", "result = i", ""])
    result = decode(code, ONE_GROUP_CIPHER)
    binding = next(binding for binding in result.bindings if binding.name == "i")
    assert [(occ.span.line, occ.span.column) for occ in binding.occurrences] == [(2, 0), (3, 9)]


def test_normalized_identifier_spans_cover_original_multibyte_spelling() -> None:
    result = absent("def e\u0301(K):\n    return K\n")
    assert [binding.name for binding in result.bindings] == ["i", "é", "K"]
    assert result.bindings[1].occurrences[0].span == SourceSpan(line=2, column=4, end_line=2, end_column=7)
    assert result.bindings[2].occurrences[0].span == SourceSpan(line=2, column=8, end_line=2, end_column=11)


@pytest.mark.parametrize("expression", ["[j for j in values]", "{j for j in values}", "{j: j for j in values}", "(j for j in values)"])
def test_all_comprehension_forms_isolate_targets_and_evaluate_first_iterable_outside(expression: str) -> None:
    result = absent(f"j = 2\nvalues = [j]\nresult = {expression}\n")
    bindings = [binding for binding in result.bindings if binding.name == "j"]
    assert len(bindings) == 2
    assert [occ.span.line for occ in bindings[0].occurrences] == [2, 3]
    assert all(occ.span.line == 4 for occ in bindings[1].occurrences)
    assert bindings[0].scope_id != bindings[1].scope_id


def test_nested_comprehension_can_capture_outer_target_without_merging_inner_target() -> None:
    result = absent("values = [[j + k for k in range(j)] for j in range(2)]\n")
    bindings = {binding.name: binding for binding in result.bindings}
    assert len(bindings["j"].occurrences) == 3
    assert len(bindings["k"].occurrences) == 2
    assert bindings["j"].scope_id != bindings["k"].scope_id


def test_async_parameters_targets_and_call_keyword_labels() -> None:
    result = absent("async def f(j, /, *args, option=1, **kwargs):\n    async with manager() as item:\n        async for value in args:\n            await g(j=j, item=item)\n")
    bindings = {binding.name: binding for binding in result.bindings}
    assert set(bindings) == {"i", "f", "j", "args", "option", "kwargs", "item", "value"}
    assert len(bindings["j"].occurrences) == 2  # The call's keyword label is excluded.
    assert [occ.kind for occ in bindings["item"].occurrences] == ["write", "read"]


def test_global_creation_and_global_barriers_redirect_free_references() -> None:
    result = absent("def outer():\n    j = 1\n    def middle():\n        global j\n        j = 2\n        def inner():\n            return j\n        return inner\n    return j\n")
    bindings = [binding for binding in result.bindings if binding.name == "j"]
    assert len(bindings) == 2
    assert [occ.span.line for occ in bindings[0].occurrences] == [3, 10]
    assert [occ.span.line for occ in bindings[1].occurrences] == [5, 6, 8]
    assert bindings[1].scope_id == "module"


def test_class_global_declaration_does_not_override_method_closure() -> None:
    result = absent("j = 0\ndef outer():\n    j = 1\n    class C:\n        global j\n        def method(self):\n            return j\n    return C\n")
    bindings = [binding for binding in result.bindings if binding.name == "j"]
    assert [occ.span.line for occ in bindings[0].occurrences] == [2, 6]
    assert [occ.span.line for occ in bindings[1].occurrences] == [4, 8]


def test_private_names_are_resolved_using_the_enclosing_class_mangling_context() -> None:
    result = absent("__item = 0\nclass C:\n    __item = 1\n    def method(self, __item):\n        return __item\n")
    bindings = [binding for binding in result.bindings if binding.name == "__item"]
    assert len(bindings) == 3
    assert [len(binding.occurrences) for binding in bindings] == [1, 1, 2]
    assert len({binding.binding_id for binding in bindings}) == 3


def test_repeated_definition_names_share_outer_binding_but_not_parameter_scopes() -> None:
    result = absent("def f(j): return j\ndef f(j): return j\n")
    functions = [binding for binding in result.bindings if binding.name == "f"]
    parameters = [binding for binding in result.bindings if binding.name == "j"]
    assert len(functions) == 1 and len(functions[0].occurrences) == 2
    assert len(parameters) == 2 and parameters[0].scope_id != parameters[1].scope_id


def test_deletion_and_same_line_exception_targets_have_exact_token_kinds() -> None:
    result = absent("j = 1\ndel j\ntry: pass\nexcept Exception as error: print(error)\n")
    bindings = {binding.name: binding for binding in result.bindings}
    assert [occ.kind for occ in bindings["j"].occurrences] == ["write", "delete"]
    assert [occ.kind for occ in bindings["error"].occurrences] == ["binding", "read"]


@pytest.mark.parametrize("code", ["type Alias = int", "def f[T](x: T): pass", "class C[T]: pass"])
def test_annotation_scopes_fail_explicitly_even_after_zero_control(code: str) -> None:
    with pytest.raises(UnsupportedSyntaxError, match="line 2"):
        absent(code)


def test_verbose_python_logging_exposes_bindings_occurrences_bits_and_frame(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="ciphers.variable_naming_in_python_v2.decoder"):
        result = absent("ordinary = i\n")
    assert result.is_encoding is False
    assert "name='i'" in caplog.text and "name='ordinary'" in caplog.text
    assert "write line=1 utf8_columns=0:1" in caplog.text
    assert "read line=2 utf8_columns=11:12" in caplog.text
    assert "bits=0 offset=0 roles=control" in caplog.text
    assert "Frame valid: control=0" in caplog.text


@pytest.fixture
def cli_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Copy fixtures into tmp_path for CLI mutation; return source and cipher paths."""
    source_file, cipher_file = tmp_path / "source.py", tmp_path / "cipher.json"
    source_file.write_bytes((ONE_GROUP / "01_basic.py").read_bytes())
    cipher_file.write_bytes((ONE_GROUP / "cipher.json").read_bytes())
    return source_file, cipher_file


def test_cli_verbose_trace_and_expect_match_preserve_json_stdout_and_logging_state(cli_inputs: tuple[Path, Path]) -> None:
    source_file, cipher_file = cli_inputs
    logger = logging.getLogger("ciphers.variable_naming_in_python_v2.decoder")
    original = (logger.handlers, logger.level, logger.propagate)
    result = CliRunner().invoke(main, [str(source_file), "--cipher", str(cipher_file), "--verbose", "--expect", "0", "--keep-only-stego-bindings"])
    assert result.exit_code == 0, result.output
    decoded = DecodedMessage.model_validate_json(result.stdout)
    assert decoded.message_bits == "0" and len(decoded.bindings) == 4
    assert "name='control'" in result.stderr  # Filtering applies to JSON, not trace collection.
    assert "utf8_columns=" in result.stderr and "roles=message" in result.stderr
    assert "Expected payload MATCH" in result.stderr
    assert (logger.handlers, logger.level, logger.propagate) == original
    quiet = CliRunner().invoke(main, [str(source_file), "--cipher", str(cipher_file)])
    assert quiet.exit_code == 0 and quiet.stderr == ""


def test_cli_expect_mismatch_is_nonzero_and_keeps_actual_decoded_result(cli_inputs: tuple[Path, Path]) -> None:
    source_file, cipher_file = cli_inputs
    result = CliRunner().invoke(main, [str(source_file), "--cipher", str(cipher_file), "--expect", "1"])
    assert result.exit_code != 0
    assert "Expected payload mismatch" in result.stderr
    assert DecodedMessage.model_validate_json(result.stdout).message_bits == "0"


@pytest.mark.parametrize(("filename", "success"), [("03_empty.py", True), ("02_absent.py", False)])
def test_cli_expect_empty_distinguishes_encoded_empty_from_absence(cli_inputs: tuple[Path, Path], filename: str, success: bool) -> None:
    source_file, cipher_file = cli_inputs
    source_file.write_bytes((ONE_GROUP / filename).read_bytes())
    result = CliRunner().invoke(main, [str(source_file), "--cipher", str(cipher_file), "--expect", ""])
    assert (result.exit_code == 0) is success


@pytest.mark.parametrize("failure", ["invalid-json", "malformed-code", "bad-expected-bits", "missing-source"])
def test_cli_input_errors_are_readable_without_tracebacks(cli_inputs: tuple[Path, Path], failure: str) -> None:
    source_file, cipher_file = cli_inputs
    arguments = [str(source_file), "--cipher", str(cipher_file)]
    if failure == "invalid-json":
        cipher_file.write_text("{", encoding="utf-8")
    elif failure == "malformed-code":
        source_file.write_text("return 1", encoding="utf-8")
    elif failure == "bad-expected-bits":
        arguments.extend(["--expect", "0x"])
    else:
        source_file.unlink()
    result = CliRunner().invoke(main, arguments)
    assert result.exit_code != 0
    assert "Error" in result.stderr and "Traceback" not in result.stderr
