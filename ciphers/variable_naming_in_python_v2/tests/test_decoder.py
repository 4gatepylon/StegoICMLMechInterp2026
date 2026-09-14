"""Executable V2 contracts, partitioned between schemas and future decoding.

Schema tests run now. Decoder tests are strict xfails only for the interface
stub's NotImplementedError; wrong results and other exceptions still fail.
Use --runxfail while implementing, then remove @needs_decoder when complete.
Fixtures are read as source, never imported or executed. Tests cover static
binding/framing behavior, not runtime semantics, dynamic exec-created names,
model quality, steganalysis, or exhaustive coverage of every Python grammar form.
"""

from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, RootModel, ValidationError

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
ONE_GROUP = FIXTURES / "codex_generated_1_group"
needs_decoder = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="Interface-only decode() stub; remove this mark when implemented")


class ExpectedCipherError(BaseModel):
    """Expected config rejection plus a positive control for the same source.

    ``error`` names the Pydantic exception; ``match`` is its expected diagnostic
    substring. ``valid_cipher_folder`` is relative to this fixture folder and
    supplies a valid cipher for ``decoded``, a complete DecodedMessage result.
    This companion check distinguishes a bad cipher from a bad encoding.
    """

    model_config = ConfigDict(extra="forbid")

    error: Literal["ValidationError"]
    match: str
    valid_cipher_folder: str
    decoded: DecodedMessage


class ExpectedDecodes(RootModel[dict[str, DecodedMessage | ExpectedCipherError]]):
    """Map each sibling Python filename to its expected decode or cipher error.

    Successful results use the decoder's public schema, including all ordinary
    and special bindings. IDs are canonicalized in first-binding order:
    binding_0, binding_1, ... and scope_0, scope_1, ... for distinct scopes.
    Repeated scope IDs express shared lexical ownership. Occurrence spans and
    all other metadata retain their original values.
    """


def load_cipher(path: Path = ONE_GROUP / "cipher.json") -> CipherConfig:
    """Validate a repo-relative JSON file as the exact cipher passed to decode."""
    return CipherConfig.model_validate_json(path.read_text(encoding="utf-8"))


def load_expectations(folder: Path) -> dict[str, DecodedMessage | ExpectedCipherError]:
    """Read a repo-relative fixture folder's manifest using ExpectedDecodes.

    Return Python-filename keys mapped to complete DecodedMessage expectations
    or ExpectedCipherError records. Discovery uses each key to find source and
    each value to select the decoding or validation-error contract.
    """
    return ExpectedDecodes.model_validate_json((folder / "expected_decodes.json").read_text(encoding="utf-8")).root


FIXTURE_FOLDERS = sorted(path.parent for path in FIXTURES.glob("*/cipher.json"))
FIXTURE_CASES = [
    pytest.param(folder / filename, expected, id=f"{folder.name}/{filename}")
    for folder in FIXTURE_FOLDERS
    for filename, expected in load_expectations(folder).items()
]


def canonicalize_ids(result: DecodedMessage) -> DecodedMessage:
    """Replace opaque IDs with first-seen labels for comparison to JSON results.

    ``result`` is a decoder response in first-occurrence order. Return the same
    DecodedMessage structure with only binding_id and scope_id relabeled. Shared
    IDs remain shared so duplicate bindings and wrong scope ownership still
    fail comparison; no source positions, contributions, or ordering change.
    """
    binding_ids: dict[str, str] = {}
    scope_ids: dict[str, str] = {}
    return result.model_copy(update={"bindings": tuple(
        binding.model_copy(update={
            "binding_id": binding_ids.setdefault(binding.binding_id, f"binding_{len(binding_ids)}"),
            "scope_id": scope_ids.setdefault(binding.scope_id, f"scope_{len(scope_ids)}"),
        })
        for binding in result.bindings
    )})


@pytest.mark.parametrize("folder", FIXTURE_FOLDERS, ids=lambda folder: folder.name)
def test_manifest_covers_exactly_the_compilable_source_files(folder: Path) -> None:
    """Every source has one schema-validated expectation; compile without running."""
    expected = load_expectations(folder)
    assert expected
    assert set(expected) == {path.name for path in folder.glob("*.py")}
    for filename in expected:
        path = folder / filename
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


@pytest.mark.parametrize(("path", "expected"), FIXTURE_CASES)
def test_fixture_cipher_validation(path: Path, expected: DecodedMessage | ExpectedCipherError) -> None:
    """Partition valid configs and explicit validation errors before decoding."""
    if isinstance(expected, ExpectedCipherError):
        with pytest.raises(ValidationError) as error:
            load_cipher(path.parent / "cipher.json")
        assert expected.match in str(error.value)
        load_cipher(path.parent / expected.valid_cipher_folder / "cipher.json")
    else:
        load_cipher(path.parent / "cipher.json")


@needs_decoder
@pytest.mark.parametrize(("path", "expected"), FIXTURE_CASES)
def test_fixture_decode(path: Path, expected: DecodedMessage | ExpectedCipherError) -> None:
    """Compare every field to JSON, then verify determinism and final filtering.

    ``path`` identifies a source file, never executed. ``expected`` comes from
    its folder's manifest. Error records select their valid companion cipher
    and positive-control decoded result; rejection is checked separately above.
    Exact comparisons cover frames, all bindings/occurrences, ownership, offsets,
    and roles. Runtime behavior and unsupported grammar partitions are omitted.
    """
    cipher_folder = path.parent
    if isinstance(expected, ExpectedCipherError):
        cipher_folder /= expected.valid_cipher_folder
        expected = expected.decoded
    cipher = load_cipher(cipher_folder / "cipher.json")
    code = path.read_text(encoding="utf-8")
    result = decode(code, cipher)
    assert canonicalize_ids(result) == expected
    assert decode(code, cipher) == result
    filtered = decode(code, cipher, keep_only_stego_bindings=True)
    assert filtered == result.model_copy(update={"bindings": special_bindings(result)})


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
def test_multibit_symbols_can_cross_every_frame_boundary() -> None:
    result = decode("d = 0\nb = 0\nc = 0\n", load_cipher(FIXTURES / "multibit_cipher.json"))
    assert result.message_bits == "11"
    assert result.length == 2
    assert [binding.bit_roles for binding in result.bindings] == [("control", "length"), ("length", "message"), ("message", "ignored")]
    assert [binding.bit_start for binding in result.bindings] == [0, 2, 4]


@needs_decoder
def test_four_length_bits_accept_fifteen_payload_bits() -> None:
    message = "001010101010101"
    result = decode(source_for_bits("1" + "1111" + message), load_cipher(FIXTURES / "four_length_bits_cipher.json"))
    assert result.message_bits == message
    assert result.length == 15


@needs_decoder
@pytest.mark.parametrize(
    ("bits", "field"),
    [("", "control"), ("1", "length"), ("100", "length"), ("10010", "payload"), ("100100", "payload")],
)
def test_incomplete_frames_raise_descriptive_exceptions(bits: str, field: str) -> None:
    with pytest.raises(IncompleteMessageError) as error:
        decode(source_for_bits(bits), load_cipher(FIXTURES / "four_length_bits_cipher.json"))
    # Exact prose is not a contract; a field name and counts make failures usable.
    text = str(error.value).lower()
    assert field in text or (field == "payload" and "message" in text)
    assert any(character.isdigit() for character in text)


@needs_decoder
@pytest.mark.parametrize("code", ["def broken(:", "return 1", "nonlocal missing", "def f():\n    nonlocal missing\n", "x = '\x00'"])
def test_parse_and_compile_errors_are_wrapped_without_executing(code: str) -> None:
    with pytest.raises(InvalidCodeError) as error:
        decode(code, load_cipher())
    assert str(error.value)
    assert isinstance(error.value.__cause__, (SyntaxError, ValueError))


@needs_decoder
def test_wildcard_import_fails_explicitly_instead_of_dropping_bindings() -> None:
    with pytest.raises(UnsupportedSyntaxError) as error:
        decode("from unavailable_dependency import *\n" + source_for_bits("1010"), load_cipher())
    assert str(error.value)


@needs_decoder
def test_no_execution_or_dependency_import_is_needed() -> None:
    code = "import unavailable_dependency\nraise RuntimeError('must not execute')\n" + source_for_bits("1010")
    assert decode(code, load_cipher()).message_bits == "0"


@needs_decoder
def test_first_occurrence_can_precede_the_binding_assignment() -> None:
    code = "def first():\n    print(j)\n    j = 1\n\ndef size(i): pass\ndef second(j): pass\ndef third(i): pass\n"
    bindings = special_bindings(decode(code, load_cipher()))
    assert [occ.kind for occ in bindings[0].occurrences] == ["read", "write"]
    assert bindings[0].occurrences[0].span.line == 2


@needs_decoder
def test_unicode_columns_are_utf8_bytes_and_strings_are_not_occurrences() -> None:
    code = 'é = "i j"; i = 0\n'
    result = decode(code, load_cipher())
    assert [binding.name for binding in result.bindings] == ["é", "i"]
    assert result.bindings[1].occurrences[0].span == SourceSpan(line=1, column=12, end_line=1, end_column=13)
