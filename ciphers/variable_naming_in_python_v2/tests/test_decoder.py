"""Executable V2 contracts, partitioned between schemas and future decoding.

Schema tests run now. Decoder tests are strict xfails only for the interface
stub's NotImplementedError; wrong results and other exceptions still fail.
Use --runxfail while implementing, then remove @needs_decoder when complete.
Fixtures are read as source, never imported or executed. Tests cover static
binding/framing behavior, not runtime semantics, dynamic exec-created names,
model quality, steganalysis, or exhaustive coverage of every Python grammar form.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ciphers.variable_naming_in_python_v2.decoder import (
    BindingOccurrence,
    CipherConfig,
    DecodedMessage,
    IncompleteMessageError,
    InvalidCodeError,
    SourceSpan,
    UnsupportedSyntaxError,
    VariableBinding,
    decode,
)

FIXTURES = Path("ciphers/variable_naming_in_python_v2/tests/fixtures")
needs_decoder = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="Interface-only decode() stub; remove this mark when implemented")
CASES = [
    pytest.param("01_basic.py", 1, "0", "110", id="positive"),
    pytest.param("02_absent.py", 4, None, "01", id="absent"),
    pytest.param("03_empty.py", 1, "", "10", id="encoded-empty"),
    pytest.param("04_maximum.py", 2, "001", "1110011", id="maximum-leading-zero-trailing"),
    pytest.param("05_reassignment.py", 1, "0", "110", id="reassignment-and-loops"),
    pytest.param("06_closures.py", 1, "0", "110", id="closures-and-shadowing"),
    pytest.param("07_parameters.py", 1, "0", "110", id="parameters-defaults-lambda"),
    pytest.param("08_namespaces.py", 1, "0", "110", id="imports-classes-definitions"),
    pytest.param("09_comprehensions.py", 1, "0", "1101", id="comprehension-walrus-global-nonlocal"),
    pytest.param("10_targets.py", 1, "0", "110", id="unpack-with-exception-match"),
]


def source(filename: str) -> str:
    """Read one repo-relative fixture as UTF-8 source for decode; execute nothing."""
    return (FIXTURES / filename).read_text(encoding="utf-8")


def starter_cipher(length_bits: int = 1) -> CipherConfig:
    """Return i=0/j=1 framing with the supplied length-field width for test calls."""
    return CipherConfig(special_variables={"index": ("i", "j")}, length_bits=length_bits)


def source_for_bits(bits: str) -> str:
    """Build independent parameter bindings for an explicitly supplied test stream.

    Each character chooses i or j in a separate function, so repeated spelling
    cannot collapse carriers. Return source only, not an encoded expected result;
    callers supply literal frame/error cases independently of this construction.
    """
    return "\n".join(f"def carrier_{index}({'j' if bit == '1' else 'i'}): pass" for index, bit in enumerate(bits))


def special_bindings(result: DecodedMessage) -> tuple[VariableBinding, ...]:
    """Select special records from a result without changing their metadata."""
    return tuple(binding for binding in result.bindings if binding.synonym_group is not None)


@pytest.mark.parametrize(
    "groups",
    [
        {},
        {"index": ()},
        {"index": ("i",)},
        {"index": ("i", "j", "k")},
        {"index": ("i", "i")},
        {"first": ("i", "j"), "second": ("j", "k")},
        {"index": ("for", "j")},
        {"index": ("not-valid", "j")},
        {"index": ("", "j")},
        {"index": ("K", "j")},
    ],
    ids=["no-groups", "empty", "singleton", "non-power-two", "duplicate-within", "duplicate-across", "keyword", "invalid-identifier", "empty-name", "nfkc-alias"],
)
def test_reject_ambiguous_or_invalid_alphabets(groups: dict[str, tuple[str, ...]]) -> None:
    with pytest.raises(ValidationError):
        CipherConfig(special_variables=groups)


@pytest.mark.parametrize("width", [0, -1, True, 1.5, "4"])
def test_reject_invalid_length_widths(width: object) -> None:
    with pytest.raises(ValidationError):
        CipherConfig(special_variables={"index": ("i", "j")}, length_bits=width)


@pytest.mark.parametrize("width", [0, 2, True, 1.0, "1"])
def test_reject_unsupported_control_widths(width: object) -> None:
    with pytest.raises(ValidationError):
        CipherConfig(special_variables={"index": ("i", "j")}, control_bits=width)


def test_alphabet_json_round_trip_preserves_symbol_order_and_multiple_widths() -> None:
    cipher = CipherConfig(special_variables={"index": ("j", "i"), "state": ("idle", "ready", "busy", "done")}, length_bits=3)
    assert CipherConfig.model_validate_json(cipher.model_dump_json()) == cipher
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CipherConfig(special_variables={"index": ("i", "j")}, length_bit=3)


@pytest.mark.parametrize(
    ("encoding", "length", "bits"),
    [(False, 0, ""), (False, None, "0"), (True, None, ""), (True, 0, None), (True, 2, "001"), (True, -1, ""), (True, 2, "0x"), (True, 2, "0\n")],
)
def test_reject_inconsistent_payload_records(encoding: bool, length: int | None, bits: str | None) -> None:
    with pytest.raises(ValidationError):
        DecodedMessage(is_encoding=encoding, length=length, message_bits=bits, bindings=())


@pytest.mark.parametrize(("encoding", "length", "bits"), [(False, None, None), (True, 0, ""), (True, 3, "001")])
def test_payload_json_round_trip_preserves_absence_empty_and_leading_zeros(encoding: bool, length: int | None, bits: str | None) -> None:
    result = DecodedMessage(is_encoding=encoding, length=length, message_bits=bits, bindings=())
    assert DecodedMessage.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(("end_line", "end_column"), [(1, 2), (1, 1), (0, 4)])
def test_reject_empty_reversed_or_invalid_spans(end_line: int, end_column: int) -> None:
    with pytest.raises(ValidationError):
        SourceSpan(line=1, column=2, end_line=end_line, end_column=end_column)


def test_binding_schema_rejects_partial_contributions_and_bad_occurrence_order() -> None:
    occurrence = BindingOccurrence(kind="write", span=SourceSpan(line=1, column=0, end_line=1, end_column=1))
    base = VariableBinding(binding_id="module:i", scope_id="module", name="i", occurrences=(occurrence,))
    invalid_records = [
        base.model_dump() | {"bits": "0"},
        base.model_dump() | {"synonym_group": "index"},
        base.model_dump() | {"synonym_group": "index", "bits": "0", "bit_start": 0, "bit_roles": ()},
        base.model_dump() | {"occurrences": (occurrence, occurrence)},
        base.model_dump() | {"occurrences": ()},
    ]
    for record in invalid_records:
        with pytest.raises(ValidationError):
            VariableBinding.model_validate(record)
    later = BindingOccurrence(kind="read", span=SourceSpan(line=2, column=0, end_line=2, end_column=1))
    with pytest.raises(ValidationError, match="source order"):
        VariableBinding.model_validate(base.model_dump() | {"occurrences": (later, occurrence)})
    symbol = VariableBinding.model_validate(base.model_dump() | {"synonym_group": "index", "bits": "01", "bit_start": 0, "bit_roles": ("control", "ignored")})
    assert VariableBinding.model_validate_json(symbol.model_dump_json()) == symbol


@needs_decoder
@pytest.mark.parametrize(("filename", "width", "message", "stream"), CASES)
def test_fixture_frames_occurrences_roles_and_final_filter(filename: str, width: int, message: str | None, stream: str) -> None:
    code = source(filename)
    result = decode(code, starter_cipher(width))
    assert result.is_encoding is (message is not None)
    assert result.message_bits == message
    assert result.length == (None if message is None else len(message))
    special = special_bindings(result)
    assert "".join(binding.bits for binding in special) == stream
    assert [binding.name for binding in special] == ["j" if bit == "1" else "i" for bit in stream]
    assert [binding.bit_start for binding in special] == list(range(len(stream)))
    roles = ["control"]
    if message is not None:
        roles.extend(["length"] * width + ["message"] * len(message))
    roles.extend(["ignored"] * (len(stream) - len(roles)))
    assert [role for binding in special for role in binding.bit_roles] == roles
    assert len({binding.binding_id for binding in result.bindings}) == len(result.bindings)
    anchors = [(binding.occurrences[0].span.line, binding.occurrences[0].span.column) for binding in result.bindings]
    assert anchors == sorted(anchors)
    lines = code.splitlines()
    for binding in result.bindings:
        for occurrence in binding.occurrences:
            span = occurrence.span
            assert span.line == span.end_line
            assert lines[span.line - 1].encode("utf-8")[span.column : span.end_column].decode("utf-8") == binding.name
    filtered = decode(code, starter_cipher(width), keep_only_stego_bindings=True)
    assert filtered == result.model_copy(update={"bindings": special})
    assert decode(code, starter_cipher(width)) == result


@needs_decoder
def test_all_ordinary_bindings_remain_in_the_default_result() -> None:
    result = decode(source("01_basic.py"), starter_cipher())
    assert [binding.name for binding in result.bindings] == ["control", "j", "size", "j", "solve", "value", "i"]
    assert [len(binding.occurrences) for binding in result.bindings] == [1, 2, 1, 2, 1, 2, 2]
    assert result.bindings[-1].occurrences[0].span == SourceSpan(line=15, column=4, end_line=15, end_column=5)


@needs_decoder
def test_repeated_assignments_and_loops_share_the_function_binding() -> None:
    bindings = special_bindings(decode(source("05_reassignment.py"), starter_cipher()))
    assert [[occ.kind for occ in binding.occurrences] for binding in bindings] == [
        ["write", "read_write", "read"],
        ["write", "write", "read", "read"],
        ["write", "read_write", "read"],
    ]
    assert len({binding.scope_id for binding in bindings}) == 3


@needs_decoder
def test_closure_reads_resolve_to_outer_binding_and_shadowing_stays_separate() -> None:
    bindings = special_bindings(decode(source("06_closures.py"), starter_cipher()))
    assert [[occ.span.line for occ in binding.occurrences] for binding in bindings] == [[5, 8], [11, 12], [18, 19]]
    assert bindings[0].binding_id != bindings[1].binding_id
    assert bindings[0].scope_id != bindings[1].scope_id


@needs_decoder
def test_defaults_resolve_outside_function_but_lambda_captures_parameter() -> None:
    bindings = special_bindings(decode(source("07_parameters.py"), starter_cipher()))
    positions = [[(occ.span.line, occ.span.column) for occ in binding.occurrences] for binding in bindings]
    assert positions == [[(3, 0), (6, 16)], [(6, 14), (7, 26)], [(7, 19), (7, 22)]]
    assert len({binding.scope_id for binding in bindings}) == 3


@needs_decoder
def test_class_namespace_does_not_capture_method_globals_or_attribute_names() -> None:
    result = decode(source("08_namespaces.py"), starter_cipher())
    bindings = special_bindings(result)
    assert [[occ.span.line for occ in binding.occurrences] for binding in bindings] == [[3, 12], [8], [15]]
    assert [binding.occurrences[0].kind for binding in bindings] == ["binding", "write", "binding"]
    assert {binding.name for binding in result.bindings} == {"j", "os", "Box", "method", "self", "i"}


@needs_decoder
def test_comprehension_walrus_and_scope_declarations_have_correct_owners() -> None:
    bindings = special_bindings(decode(source("09_comprehensions.py"), starter_cipher()))
    assert [[occ.span.line for occ in binding.occurrences] for binding in bindings] == [[3, 14, 15], [7, 10, 11, 18], [17, 18], [17, 17]]
    assert bindings[1].scope_id == bindings[2].scope_id
    assert bindings[3].scope_id != bindings[1].scope_id
    assert bindings[0].occurrences[1].kind == "declaration"
    assert bindings[1].occurrences[1].kind == "declaration"


@needs_decoder
def test_exception_and_match_captures_reuse_the_same_function_local() -> None:
    result = decode(source("10_targets.py"), starter_cipher())
    bindings = special_bindings(result)
    assert [[occ.span.line for occ in binding.occurrences] for binding in bindings] == [[3], [7, 8], [11, 12, 14, 15]]
    assert [occ.kind for occ in bindings[2].occurrences] == ["binding", "read", "binding", "read"]
    assert {binding.name for binding in result.bindings} == {"j", "ordinary", "process", "manager", "subject", "result", "i", "rest"}


@needs_decoder
def test_multibit_symbols_can_cross_every_frame_boundary() -> None:
    result = decode("d = 0\nb = 0\nc = 0\n", CipherConfig(special_variables={"state": ("a", "b", "c", "d")}, length_bits=2))
    assert result.message_bits == "11"
    assert result.length == 2
    assert [binding.bit_roles for binding in result.bindings] == [("control", "length"), ("length", "message"), ("message", "ignored")]
    assert [binding.bit_start for binding in result.bindings] == [0, 2, 4]


@needs_decoder
def test_four_length_bits_accept_fifteen_payload_bits() -> None:
    message = "001010101010101"
    result = decode(source_for_bits("1" + "1111" + message), starter_cipher(4))
    assert result.message_bits == message
    assert result.length == 15


@needs_decoder
@pytest.mark.parametrize(
    ("bits", "width", "field"),
    [("", 4, "control"), ("1", 4, "length"), ("100", 4, "length"), ("10010", 4, "payload"), ("100100", 4, "payload")],
)
def test_incomplete_frames_raise_descriptive_exceptions(bits: str, width: int, field: str) -> None:
    with pytest.raises(IncompleteMessageError) as error:
        decode(source_for_bits(bits), starter_cipher(width))
    # Exact prose is not a contract; a field name and counts make failures usable.
    text = str(error.value).lower()
    assert field in text or (field == "payload" and "message" in text)
    assert any(character.isdigit() for character in text)


@needs_decoder
@pytest.mark.parametrize("code", ["def broken(:", "return 1", "nonlocal missing", "def f():\n    nonlocal missing\n", "x = '\x00'"])
def test_parse_and_compile_errors_are_wrapped_without_executing(code: str) -> None:
    with pytest.raises(InvalidCodeError) as error:
        decode(code, starter_cipher())
    assert str(error.value)
    assert isinstance(error.value.__cause__, (SyntaxError, ValueError))


@needs_decoder
def test_wildcard_import_fails_explicitly_instead_of_dropping_bindings() -> None:
    with pytest.raises(UnsupportedSyntaxError) as error:
        decode("from unavailable_dependency import *\n" + source_for_bits("110"), starter_cipher())
    assert str(error.value)


@needs_decoder
def test_no_execution_or_dependency_import_is_needed() -> None:
    code = "import unavailable_dependency\nraise RuntimeError('must not execute')\n" + source_for_bits("110")
    assert decode(code, starter_cipher()).message_bits == "0"


@needs_decoder
def test_first_occurrence_can_precede_the_binding_assignment() -> None:
    code = "def first():\n    print(j)\n    j = 1\n\ndef second(j): pass\ndef third(i): pass\n"
    bindings = special_bindings(decode(code, starter_cipher()))
    assert [occ.kind for occ in bindings[0].occurrences] == ["read", "write"]
    assert bindings[0].occurrences[0].span.line == 2


@needs_decoder
def test_unicode_columns_are_utf8_bytes_and_strings_are_not_occurrences() -> None:
    code = 'é = "i j"; i = 0\n'
    result = decode(code, starter_cipher())
    assert [binding.name for binding in result.bindings] == ["é", "i"]
    assert result.bindings[1].occurrences[0].span == SourceSpan(line=1, column=12, end_line=1, end_column=13)
