"""Public models and static decoder for the V2 binding-based variable-name cipher.

Import ``CipherConfig``, ``DecodedMessage``, and ``decode`` from this module.
Debug logging describes the complete binding/occurrence trace and parsed frame.
Source is compiled for validation but never executed.
"""

import ast
import keyword
import logging
import unicodedata
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BinaryBits = Annotated[str, Field(pattern=r"^[01]*$")]
BitRole = Literal["control", "length", "message", "ignored"]
OccurrenceKind = Literal["binding", "read", "write", "read_write", "delete", "declaration"]
logger = logging.getLogger(__name__)


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

    ``kind`` is binding for parameters/import aliases/definition names/exception
    targets/pattern captures, write for assignment/with/loop targets, read for
    loads, read_write for augmented assignment, delete for del, and declaration
    for global/nonlocal.
    A token appears once even when it both reads and writes. ``span`` identifies
    its location; occurrences from nested scopes still reference the owning
    binding, unless shadowing creates a different binding.
    """

    model_config = ConfigDict(extra="forbid")

    kind: OccurrenceKind
    span: SourceSpan

    @model_validator(mode="after")
    def validate_single_line(self) -> Self:
        """Reject multiline identifier occurrences; return this validated token."""
        if self.span.line != self.span.end_line:
            raise ValueError("An identifier occurrence must span exactly one physical line")
        return self


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
    special bindings if the caller requested filtering. The decoder includes
    ordinary bindings, parameters, imports, definition names, and class body
    bindings, and resolves closures, comprehensions, and global/nonlocal.
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
    """Decode one Python source string under a shared cipher without executing it.

    Args:
        code: Complete, single-file Python source. Parse and compile it under
            the running Python version without executing
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
        InvalidCodeError: Malformed or non-compiling
            source, including compiler scope errors; chain the original error.
        UnsupportedSyntaxError: A construct cannot be resolved faithfully,
            including wildcard imports and PEP 695 type-parameter/alias scopes.
            Fail instead of returning silently incomplete binding data.
        IncompleteMessageError: Missing control or fewer
            length/payload bits than required by the frame.

    Invalid cipher data raises pydantic.ValidationError when constructing
    CipherConfig or revalidating its snapshot on entry. Successful compilation
    does not establish runtime correctness or dependency availability. Resolution
    is lexical: it does not simulate class-local runtime fallback or dynamic name
    creation. Enable DEBUG on this module's logger to inspect bindings and frames.
    """
    # Extraction consumes the public record types in this module; defer the
    # import so they are available before bindings imports them.
    from ciphers.variable_naming_in_python_v2.bindings import collect_bindings

    cipher = CipherConfig.model_validate(cipher.model_dump())
    try:
        tree = ast.parse(code, filename="<stego-source>", mode="exec")
        compile(tree, "<stego-source>", "exec", dont_inherit=True)
    except (SyntaxError, ValueError, RecursionError) as error:
        logger.debug("Source validation failed: %s", error)
        raise InvalidCodeError(f"Python source cannot compile: {error}") from error
    bindings = collect_bindings(code, tree)
    alphabet = {name: (group, format(index, f"0{(len(names) - 1).bit_length()}b")) for group, names in cipher.special_variables.items() for index, name in enumerate(names)}
    stream = "".join(alphabet[binding.name][1] for binding in bindings if binding.name in alphabet)
    # Log collection before framing so truncated inputs still have a useful trace.
    if logger.isEnabledFor(logging.DEBUG):
        lines = code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        for binding in bindings:
            logger.debug("Binding name=%r id=%s scope=%s symbol=%r", binding.name, binding.binding_id, binding.scope_id, alphabet.get(binding.name))
            for occurrence in binding.occurrences:
                span = occurrence.span
                logger.debug("  %s line=%d utf8_columns=%d:%d source=%r", occurrence.kind, span.line, span.column, span.end_column, lines[span.line - 1])
    length, message_bits, roles = _parse_frame(stream, cipher.length_bits)
    enriched = []
    offset = 0
    for binding in bindings:
        if binding.name in alphabet:
            group, bits = alphabet[binding.name]
            contribution_roles = roles[offset : offset + len(bits)]
            binding = VariableBinding(
                binding_id=binding.binding_id,
                scope_id=binding.scope_id,
                name=binding.name,
                occurrences=binding.occurrences,
                synonym_group=group,
                bits=bits,
                bit_start=offset,
                bit_roles=contribution_roles,
            )
            logger.debug("Symbol id=%s bits=%s offset=%d roles=%s", binding.binding_id, bits, offset, ",".join(contribution_roles))
            offset += len(bits)
        enriched.append(binding)
    return DecodedMessage(
        is_encoding=length is not None,
        length=length,
        message_bits=message_bits,
        bindings=tuple(binding for binding in enriched if not keep_only_stego_bindings or binding.synonym_group is not None),
    )


def _parse_frame(stream: str, length_bits: int) -> tuple[int | None, str | None, tuple[BitRole, ...]]:
    """Parse a collected bit stream; return length, exact payload, and per-bit roles.

    ``stream`` contains only binary digits from the validated alphabet;
    ``length_bits`` is a positive validated header width. Roles cover the entire
    stream, including ignored trailing bits, so callers can slice a multi-bit
    symbol across field boundaries. Absent frames return None/None. Raise
    IncompleteMessageError with the field and available/required counts when a
    control, length, or payload field is truncated. Never allocate by a declared
    payload length until checking that those bits are actually available.
    """
    if not stream:
        raise IncompleteMessageError("Missing control: need 1 bit, found 0")
    if stream[0] == "0":
        logger.debug("Frame valid: control=0 encoding=False length=None message=None ignored_bits=%s", stream[1:])
        return None, None, ("control",) + ("ignored",) * (len(stream) - 1)
    header_end = 1 + length_bits
    if len(stream) < header_end:
        raise IncompleteMessageError(f"Truncated length: need {length_bits} bits, found {len(stream) - 1}")
    length = int(stream[1:header_end], 2)
    if len(stream) - header_end < length:
        raise IncompleteMessageError(f"Truncated payload: declared {length} bits, found {len(stream) - header_end}")
    payload_end = header_end + length
    payload = stream[header_end:payload_end]
    roles: tuple[BitRole, ...] = ("control",) + ("length",) * length_bits + ("message",) * length + ("ignored",) * (len(stream) - payload_end)
    logger.debug("Frame valid: control=1 length_bits=%s length=%d message=%s ignored_bits=%s", stream[1:header_end], length, payload, stream[payload_end:])
    return length, payload, roles
