"""Behavioral tests for the repo-local share extractor's stdin/JSON interface.

Partitions: a complete graph versus missing/cyclic/reordered/unfinished data;
visible text versus hidden/context/reasoning/multimodal/redacted content; valid
reference streams versus missing/truncated/unknown/cyclic encodings. Fixtures are
synthetic, never downloaded conversations. These tests omit HTTP access, browser
rendering, attachment retrieval, and compatibility with unobserved HTML formats.
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = ".agents/skills/read-chatgpt-share/scripts/extract.py"


def node(node_id: str, parent: str, role: str, text: str, **metadata: object) -> dict:
    """Build a synthetic text-message node with caller-selected identity and role.

    ``parent`` identifies its predecessor; ``text`` becomes its sole content part.
    Additional keyword arguments populate message metadata, allowing tests to vary
    visibility, redaction, and completion. Return a node dictionary with id,
    parent, children, and message fields matching the downloaded share schema.
    """
    return {
        "id": node_id,
        "parent": parent,
        "children": [],
        "message": {
            "id": node_id,
            "author": {"role": role},
            "content": {"content_type": "text", "parts": [text]},
            "channel": "final" if role == "assistant" else None,
            "status": "finished_successfully",
            "end_turn": role == "assistant",
            "metadata": metadata,
        },
    }


@pytest.fixture
def conversation() -> dict:
    """Supply one root and one completed exchange, with consistent parent/child links."""
    root = {"id": "root", "children": ["u"]}
    user = node("u", "root", "user", "Explain λ and 日本語.\n\n```python\nx = 1\n```")
    user["children"] = ["a"]
    assistant = node("a", "u", "assistant", "Preserve  spaces\nand newlines.\n", is_complete=True)
    return {
        "title": "Synthetic share",
        "is_public": True,
        "mapping": {item["id"]: item for item in [root, user, assistant]},
        "linear_conversation": [root, user, assistant],
        "current_node": "a",
    }


def stream_table(conversation: dict) -> list:
    """Encode synthetic route data using indexed keys/values and the observed null.

    ``conversation`` is the graph fixture consumed as serverResponse.data. Return
    a JSON-serializable reference table, rooted at loaderData. Unrelated root data
    deliberately contains a deferred promise to check that only the share route
    is traversed. Each fixture entry gets its own index; no application encoder
    or production extractor code is called to generate expected output.
    """
    values = []

    def encode(value: object) -> int:
        if value is None:
            return -5
        index = len(values)
        values.append(None)
        if isinstance(value, dict):
            values[index] = {f"_{encode(key)}": encode(item) for key, item in value.items()}
        elif isinstance(value, list):
            values[index] = [encode(item) for item in value]
        else:
            values[index] = value
        return index

    encode(
        {
            "loaderData": {
                "root": {"irrelevant": "DEFERRED"},
                "routes/share.$shareId.($action)": {
                    "sharedConversationId": "00000000-0000-4000-8000-000000000001",
                    "serverResponse": {"type": "data", "data": conversation},
                },
            }
        }
    )
    values[values.index("DEFERRED")] = ["P", 999]
    return values


def html_page(table: list, *, closed: bool = True, split: bool = False) -> str:
    """Wrap a reference table in stream scripts plus unrelated bootstrap noise.

    ``closed`` controls the stream-close marker; ``split`` divides the initial
    table across enqueue calls to exercise string-chunk assembly. Return HTML
    with a later unrelated promise record, like the observed public share page.
    """
    stream = json.dumps(table, ensure_ascii=True)
    chunks = [stream[: len(stream) // 2], stream[len(stream) // 2 :]] if split else [stream]
    chunks.append("\nP999:[{}]\n")
    scripts = [f"<script>window.__reactRouterContext.streamController.enqueue({json.dumps(chunk)});</script>" for chunk in chunks]
    if closed:
        scripts.append("<script>window.__reactRouterContext.streamController.close();</script>")
    return '<html><script id="client-bootstrap" type="application/json">{"noise":"do not include me"}</script>' + "".join(scripts) + "</html>"


def run_extractor(html: str, *args: str) -> subprocess.CompletedProcess:
    """Exercise the real Click CLI using HTML stdin and capture both output streams.

    ``args`` are optional CLI switches; the subprocess uses the current test
    interpreter (stego) and repo-root cwd. Return CompletedProcess so tests can
    verify exit status, parse stdout JSON, and inspect error hygiene on stderr.
    """
    return subprocess.run([sys.executable, EXTRACTOR, *args], input=html, capture_output=True, text=True, cwd=REPO_ROOT, timeout=20)


@pytest.mark.parametrize("split", [False, True])
def test_preserves_message_text_and_checks_complete_graph(conversation: dict, split: bool) -> None:
    result = run_extractor(html_page(stream_table(conversation), split=split))
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert [message["parts"] for message in output["messages"]] == [["Explain λ and 日本語.\n\n```python\nx = 1\n```"], ["Preserve  spaces\nand newlines.\n"]]
    assert output["report"]["chain_matches_linear"]
    assert output["report"]["linear_matches_mapping"]
    assert output["report"]["final_assistant_marked_complete"]
    assert output["report"]["warnings"] == []
    assert "do not include me" not in result.stdout
    assert "DEFERRED" not in result.stdout


@pytest.mark.parametrize("kind", ["hidden", "context", "reasoning", "analysis", "redacted", "multimodal", "attachment"])
def test_filters_non_transcript_records_and_reports_omissions(conversation: dict, kind: str) -> None:
    message = conversation["linear_conversation"][1]["message"]
    marker = "UNTRUSTED_PAYLOAD_MARKER"
    message["content"]["parts"] = [marker]
    if kind == "hidden":
        message["metadata"]["is_visually_hidden_from_conversation"] = True
    elif kind == "context":
        message["author"]["role"] = "system"
    elif kind == "reasoning":
        message["content"] = {"content_type": "thoughts", "thoughts": [{"text": marker}]}
    elif kind == "analysis":
        message["author"]["role"] = "assistant"
        message["channel"] = "analysis"
    elif kind == "redacted":
        message["metadata"]["is_redacted"] = True
    elif kind == "multimodal":
        message["content"] = {"content_type": "multimodal_text", "parts": ["Describe this image.", {"image_asset_pointer": marker}]}
    else:
        message["content"]["parts"] = ["Read the attached file."]
        message["metadata"]["attachments"] = [{"name": marker}]
    result = run_extractor(html_page(stream_table(conversation)))
    assert result.returncode == 0, result.stderr
    assert marker not in result.stdout
    output = json.loads(result.stdout)
    assert any(item["node_id"] == "u" for item in output["report"]["omissions"])
    if kind in ("multimodal", "attachment"):
        assert len(output["messages"]) == 2
        assert output["report"]["warnings"]
    else:
        assert [message["role"] for message in output["messages"]] == ["assistant"]


@pytest.mark.parametrize("damage", ["missing_parent", "cycle", "reordered", "record_mismatch", "unfinished", "no_close"])
def test_reports_incomplete_or_inconsistent_snapshots(conversation: dict, damage: str) -> None:
    if damage == "missing_parent":
        conversation["mapping"]["a"]["parent"] = "missing"
    elif damage == "cycle":
        conversation["mapping"]["a"]["parent"] = "a"
    elif damage == "reordered":
        conversation["linear_conversation"] = list(reversed(conversation["linear_conversation"]))
    elif damage == "record_mismatch":
        conversation["mapping"] = copy.deepcopy(conversation["mapping"])
        conversation["mapping"]["a"]["message"]["content"]["parts"] = ["Different graph text"]
    elif damage == "unfinished":
        conversation["mapping"]["a"]["message"]["status"] = "in_progress"
    output = json.loads(run_extractor(html_page(stream_table(conversation), closed=damage != "no_close")).stdout)
    assert output["report"]["warnings"]
    if damage == "unfinished":
        assert output["report"]["unfinished_message_ids"] == ["a"]
        assert not output["report"]["final_assistant_marked_complete"]
    elif damage == "record_mismatch":
        assert not output["report"]["linear_matches_mapping"]
    elif damage == "no_close":
        assert not output["report"]["stream_closed"]
    else:
        assert not output["report"]["chain_matches_linear"]


@pytest.mark.parametrize("damage", ["missing", "truncated", "tagged", "reference_cycle", "out_of_range", "unknown_sentinel", "schema"])
def test_rejects_unsupported_pages_without_emitting_payload(conversation: dict, damage: str) -> None:
    marker = "MUST_NOT_APPEAR_IN_ERRORS"
    conversation["title"] = marker
    table = stream_table(conversation)
    if damage == "tagged":
        table[table.index(marker)] = ["UnknownEncoding", 0]
    elif damage == "reference_cycle":
        index = table.index(marker)
        table[index] = [index]
    elif damage == "out_of_range":
        table[table.index(marker)] = [len(table) + 100]
    elif damage == "unknown_sentinel":
        table[table.index(marker)] = [-999]
    elif damage == "schema":
        index = table.index(marker)
        table[index] = [len(table)]
        table.append(marker)
    html = html_page(table)
    if damage == "missing":
        html = "<html>Sign in to continue</html>"
    elif damage == "truncated":
        html = html[: len(html) // 2]
    result = run_extractor(html)
    assert result.returncode != 0
    assert result.stdout == ""
    assert marker not in result.stderr


def test_report_only_does_not_expose_message_text(conversation: dict) -> None:
    result = run_extractor(html_page(stream_table(conversation)), "--report-only")
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert "messages" not in output
    assert "Preserve  spaces" not in result.stdout
    assert output["report"]["text_characters"] == sum(
        len(part) for node in conversation["linear_conversation"] if "message" in node for part in node["message"]["content"]["parts"]
    )
