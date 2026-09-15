"""Run one APPS evaluation inside a Modal Sandbox.

The local orchestrator uploads this file, ``request.json``, and the verified
annotated ``apps_evaluator.py`` under ``STEGO_ARTIFACTS_DIR``. This process must
not import repository packages: the image supplies only runtime dependencies.

``request.json`` requires ``code`` (source text), ``input_output`` (paired
inputs/outputs and nullable ``fn_name``), ``case_timeout_s`` (integer alarm
limit), and ``max_log_chars`` (positive diagnostic length). Stdout is a single
JSON object with ``results``, ``logs``, and ``error``. The APPS reliability
guard mutates process globals, so one process is used per solution.
"""

from __future__ import annotations

import contextlib
import faulthandler
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _dbg(debug_lines: list[str], message: str) -> None:
    # APPS may replace sys.stderr with StringIO; this list survives that.
    debug_lines.append(message)


def _describe_stream(debug_lines: list[str], name: str, stream: object) -> None:
    try:
        fileno = getattr(stream, "fileno")()
        _dbg(
            debug_lines,
            f"{name}: type={type(stream).__name__} module={type(stream).__module__} fileno={fileno}",
        )
    except Exception as exc:
        _dbg(
            debug_lines,
            f"{name}: type={type(stream).__name__} module={type(stream).__module__} "
            f"fileno raised {type(exc).__name__}: {exc}",
        )


def _install_faulthandler_hook(debug_lines: list[str]) -> None:
    original_enable = faulthandler.enable

    def logged_enable(*args: Any, **kwargs: Any) -> None:
        target = kwargs.get("file", args[0] if args else sys.stderr)
        _dbg(debug_lines, f"faulthandler.enable args={args!r} kwargs={kwargs!r}")
        _describe_stream(debug_lines, "faulthandler target", target)
        _describe_stream(debug_lines, "sys.stderr at enable", sys.stderr)
        _describe_stream(debug_lines, "sys.__stderr__ at enable", sys.__stderr__)
        try:
            original_enable(*args, **kwargs)
        except Exception as exc:
            _dbg(debug_lines, f"faulthandler.enable raised {type(exc).__name__}: {exc}")
            _dbg(debug_lines, "".join(traceback.format_stack()))
            raise

    faulthandler.enable = logged_enable


def _load_evaluator(root: Path, case_timeout_s: int) -> Any:
    spec = importlib.util.spec_from_file_location("apps_evaluator", root / "apps_evaluator.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load APPS evaluator from {root / 'apps_evaluator.py'}")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    evaluator.timeout = case_timeout_s
    return evaluator


def _record_evaluator_source_hints(debug_lines: list[str], source: str) -> None:
    tokens = ("faulthandler", "swallow_io", "WriteOnlyStringIO", "redirect_stderr", "fileno")
    for index, line in enumerate(source.splitlines(), start=1):
        if any(token in line for token in tokens):
            _dbg(debug_lines, f"apps_evaluator.py:{index}: {line.rstrip()}")


def _format_error(exception: BaseException, debug_lines: list[str]) -> str:
    return (
        f"{type(exception).__name__}: {exception}\n"
        f"{traceback.format_exc()}\n"
        f"--- driver debug ---\n" + "\n".join(debug_lines)
    )


def _run_with_log_capture(
    root: Path,
    payload: dict[str, Any],
    evaluator: Any,
    debug_lines: list[str],
) -> tuple[list[Any], str | None, str]:
    results: list[Any] = []
    error: str | None = None
    # APPS enables faulthandler against sys.stderr; StringIO has no usable fileno.
    # Open the capture file before APPS's reliability guard changes process globals.
    log_path = root / "evaluator.log"
    with log_path.open("w+", encoding="utf-8") as logs:
        _describe_stream(debug_lines, "evaluator.log", logs)
        with contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
            _describe_stream(debug_lines, "sys.stderr after redirect", sys.stderr)
            try:
                raw = evaluator.run_test(
                    problem={"input_output": payload["input_output"]},
                    test=payload["code"],
                    debug=True,
                )
                results = [value.item() if hasattr(value, "item") else value for value in raw]
            except BaseException as exception:
                error = _format_error(exception, debug_lines)
        logs.flush()
        logs.seek(0)
        captured = logs.read()
    prefix = ""
    if debug_lines:
        prefix = "--- driver debug ---\n" + "\n".join(debug_lines) + "\n--- captured ---\n"
    diagnostic_tail = (prefix + captured)[-payload["max_log_chars"] :]
    return results, error, diagnostic_tail


def main() -> None:
    """Load ``request.json``, run the pinned evaluator, and print one JSON verdict.

    Reads ``STEGO_ARTIFACTS_DIR/request.json`` and ``apps_evaluator.py``. Writes
    ``evaluator.log`` for captured stdio. Prints ``results``, ``logs``, and
    ``error`` as JSON on stdout for the local orchestrator to parse.
    """
    root = Path(os.environ["STEGO_ARTIFACTS_DIR"])
    payload = json.loads((root / "request.json").read_text())
    debug_lines: list[str] = []
    _install_faulthandler_hook(debug_lines)
    evaluator = _load_evaluator(root, payload["case_timeout_s"])
    _record_evaluator_source_hints(debug_lines, (root / "apps_evaluator.py").read_text())
    _describe_stream(debug_lines, "sys.stderr before redirect", sys.stderr)
    _describe_stream(debug_lines, "sys.__stderr__ before redirect", sys.__stderr__)
    results, error, diagnostic_tail = _run_with_log_capture(root, payload, evaluator, debug_lines)
    print(json.dumps({"results": results, "logs": diagnostic_tail, "error": error}))


if __name__ == "__main__":
    main()
