"""Public contracts for the V2 binding-based decoder; extraction is not implemented.

Import ``CipherConfig``, ``DecodedMessage``, and ``decode`` from this module.
The models are usable now so callers and a future decoder can develop against
the same schema. ``decode`` deliberately raises ``NotImplementedError``.
"""

import keyword
import unicodedata
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BinaryBits = Annotated[str, Field(pattern=r"^[01]*$")]
BitRole = Literal["control", "length", "message", "ignored"]
OccurrenceKind = Literal["binding", "read", "write", "read_write", "delete", "declaration"]


class CipherConfig(BaseModel):
    """Describe the name-to-bits alphabet and frame shared by encoder and decoder.

    ``special_variables`` maps descriptive group labels to ordered name tuples.
    Labels do not influence decoding. Each tuple has 2**K distinct names, K >= 1;
    a name emits its zero-padded K-bit tuple index, most significant bit first.
    Names must be globally unique, valid normalized Python identifiers, and not
    keywords. Normalization prevents distinct config spellings referring to the
    same identifier after Python parses source.

    ``control_bits`` is fixed at one: zero means absent, one means present.
    ``length_bits`` is the positive width of the unsigned, big-endian payload
    length field. An encoded payload of length L must satisfy
    0 <= L < 2**length_bits, equivalently L.bit_length() <= length_bits.
    The decoder must enforce frame completeness; this config has no payload to
    validate. Zero control needs neither a length field nor a payload.
    """

    model_config = ConfigDict(extra="forbid")

    special_variables: dict[str, tuple[str, ...]] = Field(min_length=1)
    control_bits: Literal[1] = 1
    length_bits: int = Field(default=4, ge=1, strict=True)

    @field_validator("control_bits", mode="before")
    @classmethod
    def validate_control_width(cls, value: object) -> object:
        """Reject non-integer widths before Literal validation; return the input."""
        if type(value) is not int:
            raise ValueError("control_bits must be the integer 1")
        return value

    @model_validator(mode="after")
    def validate_alphabet(self) -> Self:
        """Validate group partitions for unambiguous decoding; return this config.

        Pydantic invokes this after parsing the fields. It rejects unsupported
        group widths, duplicate identifiers, keywords, and non-normalized names
        as ValidationError entries before any caller can use the alphabet.
        """
        seen: set[str] = set()
        for group, names in self.special_variables.items():
            if len(names) < 2 or len(names) & (len(names) - 1):
                raise ValueError(f"Group {group!r} must contain 2**K names with K >= 1")
            for name in names:
                if not name.isidentifier() or keyword.iskeyword(name):
                    raise ValueError(f"Invalid Python variable name: {name!r}")
                if unicodedata.normalize("NFKC", name) != name:
                    raise ValueError(f"Variable name must be NFKC-normalized: {name!r}")
                if name in seen:
                    raise ValueError(f"Variable name occurs more than once: {name!r}")
                seen.add(name)
        return self


class SourceSpan(BaseModel):
    """Half-open identifier span in the original input source, matching AST units.

    Lines are one-based; columns are zero-based UTF-8 byte offsets, not Unicode
    character indices. End coordinates are exclusive. Occurrence spans cover
    identifier tokens, not their containing statements or expressions.
    """

    model_config = ConfigDict(extra="forbid")

    line: int = Field(ge=1, strict=True)
    column: int = Field(ge=0, strict=True)
    end_line: int = Field(ge=1, strict=True)
    end_column: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        """Reject empty/reversed spans; return the validated source coordinates."""
        if (self.end_line, self.end_column) <= (self.line, self.column):
            raise ValueError("Source span end must follow its start")
        return self


class BindingOccurrence(BaseModel):
    """One identifier token associated with a lexical binding.

    ``kind`` is binding for parameters/import aliases/definition names/pattern
    captures, write for assignment or loop targets, read for loads, read_write
    for augmented assignment, delete for del, and declaration for global/nonlocal.
    A token appears once even when it both reads and writes. ``span`` identifies
    its location; occurrences from nested scopes still reference the owning
    binding, unless shadowing creates a different binding.
    """

    model_config = ConfigDict(extra="forbid")

    kind: OccurrenceKind
    span: SourceSpan


class VariableBinding(BaseModel):
    """One lexical binding and its contribution to the complete decoded stream.

    ``binding_id`` and ``scope_id`` are opaque deterministic IDs for the same
    source. The producer must distinguish scopes by source structure/position,
    not merely function names. ``name`` is the normalized source identifier,
    before class-private name mangling. Reassignments keep the same binding ID.

    ``occurrences`` contains all associated identifier tokens in source order,
    including declarations and reads before assignments. Its first occurrence
    determines this binding's position in the stream. Import paths, attribute
    names, keyword argument labels, comments, and strings are not occurrences.

    Ordinary bindings have ``synonym_group=None``, ``bits=""``,
    ``bit_start=None``, and ``bit_roles=()``. Special bindings name their cipher
    group, emit ``bits`` once, and start at zero-based ``bit_start`` in the full
    stream, before truncating the frame. Each emitted bit has a matching role:
    control, length, message, or ignored. A multi-bit symbol can straddle fields.
    Ignored includes every bit after zero control or after the framed payload.

    This model checks local consistency. The decoder must verify group/index
    correctness, binding resolution, and globally contiguous bit offsets against
    its CipherConfig. Filtering ordinary bindings never changes these offsets.
    """

    model_config = ConfigDict(extra="forbid")

    binding_id: str = Field(min_length=1)
    scope_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    occurrences: tuple[BindingOccurrence, ...] = Field(min_length=1)
    synonym_group: str | None = None
    bits: BinaryBits = ""
    bit_start: int | None = Field(default=None, ge=0, strict=True)
    bit_roles: tuple[BitRole, ...] = ()

    @model_validator(mode="after")
    def validate_contribution(self) -> Self:
        """Validate occurrence order and symbol metadata; return this binding.

        Pydantic rejects duplicate/out-of-order occurrences, partial ordinary
        binding metadata, or special symbols without one role per emitted bit.
        Cross-binding and cipher-dependent checks remain the decoder's contract.
        """
        positions = [(occ.span.line, occ.span.column) for occ in self.occurrences]
        if positions != sorted(set(positions)):
            raise ValueError("Occurrences must be unique and in source order")
        if self.synonym_group is None:
            if self.bits or self.bit_start is not None or self.bit_roles:
                raise ValueError("Ordinary bindings cannot contribute bits")
        elif not self.bits or self.bit_start is None or len(self.bit_roles) != len(self.bits):
            raise ValueError("Special bindings require bits, a start offset, and one role per bit")
        return self


class DecodedMessage(BaseModel):
    """Successful decode, including an explicitly encoded absence of a message.

    ``is_encoding=False`` requires both ``length`` and ``message_bits`` to be
    None. True requires a nonnegative length equal to len(message_bits); an
    encoded empty message is True, 0, "". Bit strings preserve leading zeros.

    ``bindings`` contains all source bindings in first-occurrence order, or only
    special bindings if the caller requested filtering. The future decoder must
    include ordinary bindings, parameters, imports, definition names, and class
    body bindings, and resolve closures, comprehensions, and global/nonlocal.
    Unresolved external names and implicit compiler temporaries are not source
    bindings. Attribute lookup is excluded; dynamically created names are not
    inferred. This is static lexical resolution, not a runtime execution trace.

    The schema enforces payload/nullability consistency. The decoder additionally
    guarantees frame roles, capacity, unique binding IDs, and result ordering.
    """

    model_config = ConfigDict(extra="forbid")

    is_encoding: bool = Field(strict=True)
    length: int | None = Field(ge=0, strict=True)
    message_bits: BinaryBits | None
    bindings: tuple[VariableBinding, ...]

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        """Reject inconsistent absence/length/payload combinations; return self."""
        if not self.is_encoding:
            if self.length is not None or self.message_bits is not None:
                raise ValueError("Non-encoding results require length=None and message_bits=None")
        elif self.length is None or self.message_bits is None or self.length != len(self.message_bits):
            raise ValueError("Encoding results require length equal to the number of message bits")
        return self


class DecodeError(ValueError):
    """Base for expected decoding failures; the exception message explains why."""


class InvalidCodeError(DecodeError):
    """Source cannot parse or compile; include compiler location where available."""


class UnsupportedSyntaxError(DecodeError):
    """A source construct cannot be resolved reliably; identify it and its location.

    The decoder must fail explicitly instead of silently omitting bindings from
    unsupported constructs. This includes wildcard imports, whose bound names
    cannot be enumerated from the supplied source string alone.
    """


class IncompleteMessageError(DecodeError):
    """Insufficient control, length, or payload bits; report field and bit counts."""


def decode(code: str, cipher: CipherConfig, *, keep_only_stego_bindings: bool = False) -> DecodedMessage:
    """Decode one Python source string under a shared cipher (interface stub).

    Args:
        code: Complete, single-file Python source. The implementation must parse
            and compile it under the running Python version without executing
            code, evaluating annotations, or importing its dependencies.
        cipher: Validated alphabet and framing widths. Every special binding
            emits its fixed-width synonym index once, ordered by its earliest
            occurrence in the original source (line, then UTF-8 byte column).
        keep_only_stego_bindings: If True, filter ordinary bindings from the
            final result after decoding. This must not change the decoded frame,
            occurrence lists, IDs, bit offsets, or roles of retained bindings.

    Returns:
        DecodedMessage with is_encoding, length, message_bits, and bindings as
        documented on that model. A zero control bit returns False/None/None;
        missing control is an error. For control one, read exactly length_bits
        unsigned big-endian length bits, then that many payload bits. The length
        must fit the header: length.bit_length() <= cipher.length_bits. Reject
        truncated fields; ignore trailing bits but preserve their binding
        metadata. Include all statically identifiable source bindings, even
        unused ones and bindings in code that would not execute.

    Raises:
        NotImplementedError: Always in this interface-only version. Replace the
            stub with extraction/framing logic in the implementation PR.
        InvalidCodeError: Future implementation: malformed or non-compiling
            source, including compiler scope errors; chain the original error.
        UnsupportedSyntaxError: Future implementation: a construct cannot be
            resolved faithfully, rather than returning silently incomplete data.
        IncompleteMessageError: Future implementation: missing control or fewer
            length/payload bits than required by the frame.

    Invalid cipher data raises pydantic.ValidationError when constructing
    CipherConfig, before calling this function. Successful compilation does not
    establish runtime correctness or that imports/dependencies are available.
    """
    raise NotImplementedError("The V2 decoder interface is defined; binding extraction and decoding are not implemented yet")
