"""Click entry point for decoding source, tracing bindings, and checking a payload.

Run from the repository root with python -m
ciphers.variable_naming_in_python_v2.cli --cipher CIPHER.json [--verbose] CODE.py.
The JSON result goes to stdout; diagnostics and expected-message comparisons go
to stderr. Input files are read only, and source is never executed.
"""

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import click
from pydantic import TypeAdapter, ValidationError

from ciphers.variable_naming_in_python_v2.decoder import BinaryBits, CipherConfig, DecodeError, decode


@contextmanager
def _diagnostics(verbose: bool) -> Iterator[logging.Logger]:
    """Temporarily route decoder diagnostics to this invocation's stderr.

    ``verbose`` enables DEBUG binding/frame logs; otherwise only warnings/errors
    are emitted. Yield the same logger for CLI expected-payload checks. Restore
    previous handlers, level, and propagation afterward so repeated CliRunner
    calls or embedding applications do not accumulate handlers or retain streams.
    """
    logger = logging.getLogger("ciphers.variable_naming_in_python_v2.decoder")
    previous_handlers, previous_level, previous_propagate = logger.handlers, logger.level, logger.propagate
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    logger.propagate = False
    try:
        yield logger
    finally:
        logger.handlers, logger.level, logger.propagate = previous_handlers, previous_level, previous_propagate
        handler.close()


def _expected_bits(ctx: click.Context, parameter: click.Parameter, value: str | None) -> str | None:
    """Validate --expect through the shared Pydantic binary-string type.

    Click supplies ``ctx`` and ``parameter`` for diagnostic context; ``value`` is
    None when omitted or the literal expected payload, including an empty string.
    Return validated bits unchanged, preserving leading zeros. Invalid characters
    produce a Click parameter error before any source files are read.
    """
    if value is None:
        return None
    try:
        return TypeAdapter(BinaryBits).validate_python(value)
    except ValidationError as error:
        raise click.BadParameter("must contain only 0 and 1 (empty is allowed)", ctx=ctx, param=parameter) from error


@click.command(help="Decode a Python source file using a JSON cipher. Emit JSON on stdout; use --verbose for the binding trace and --expect to verify the secret.")
@click.argument("source_file", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path))
@click.option("--cipher", "cipher_file", required=True, type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path), help="JSON CipherConfig file.")
@click.option("--verbose", "-v", is_flag=True, help="Show all bindings, occurrence locations, emitted bits, and frame checks on stderr.")
@click.option("--expect", callback=_expected_bits, help="Require an encoded message with these exact bits; mismatches exit nonzero.")
@click.option("--keep-only-stego-bindings", is_flag=True, help="Filter ordinary bindings from the final JSON, after decoding.")
def main(source_file: Path, cipher_file: Path, verbose: bool, expect: str | None, keep_only_stego_bindings: bool) -> None:
    """Decode SOURCE_FILE using a JSON cipher and optionally verify expected bits.

    ``source_file`` and ``cipher_file`` are CLI paths, normally relative to the
    repository root; generated inputs can be supplied from STEGO_ARTIFACTS_DIR.
    ``verbose`` enables the complete trace, including ordinary bindings even if
    ``keep_only_stego_bindings`` filters the final JSON. ``expect=None`` disables
    semantic comparison; an empty string requires an encoded zero-length payload,
    not an absent message. Successful framing alone cannot establish the intended
    secret; only the explicit expected comparison does that.

    Return nothing. Emit one DecodedMessage JSON document to stdout after a
    successful decode, even if its expected-message comparison subsequently
    fails. ClickException gives read/config/decode errors or payload mismatches a
    readable message and nonzero exit without hiding unexpected programming errors.
    """
    with _diagnostics(verbose) as logger:
        try:
            cipher = CipherConfig.model_validate_json(cipher_file.read_text(encoding="utf-8"))
            result = decode(source_file.read_text(encoding="utf-8"), cipher, keep_only_stego_bindings=keep_only_stego_bindings)
        except (OSError, UnicodeError, ValidationError, DecodeError) as error:
            raise click.ClickException(str(error)) from error
        click.echo(result.model_dump_json(indent=2))
        if expect is not None:
            if not result.is_encoding or result.message_bits != expect:
                actual = repr(result.message_bits) if result.is_encoding else "absent"
                raise click.ClickException(f"Expected payload mismatch: expected {expect!r}, decoded {actual}")
            logger.info("Expected payload MATCH: expected=%r decoded=%r", expect, result.message_bits)
        elif verbose:
            logger.info("Framing verified; no --expect supplied, so intended-message equality was not checked")


if __name__ == "__main__":
    main()
