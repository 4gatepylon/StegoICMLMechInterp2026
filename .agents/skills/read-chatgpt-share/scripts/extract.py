"""Extract a public ChatGPT share's text from downloaded HTML, without running JS.

Run from the repo root in the stego environment with HTML on stdin. JSON on stdout
contains only the conversation text and an audit report; application bootstrap
data, hidden context, and reasoning records are not included. The parser supports
the reference-table stream observed in September 2026, not arbitrary JavaScript
or every historical/future ChatGPT renderer. No network or filesystem I/O occurs.
"""

import json
import sys
from collections import Counter
from html.parser import HTMLParser
from typing import Any, Literal, override
from uuid import UUID

import click
from pydantic import BaseModel, ConfigDict, Field, ValidationError

STREAM_PREFIX = "window.__reactRouterContext.streamController.enqueue("
STREAM_CLOSE = "window.__reactRouterContext.streamController.close();"
# Bound untrusted input and recursive references before expanding the route data.
MAX_HTML_BYTES = 16 * 1024 * 1024
MAX_REFERENCE_DEPTH = 160


class PayloadModel(BaseModel):
    """Validate consumed fields strictly while ignoring unrelated web-app metadata."""

    model_config = ConfigDict(extra="ignore", strict=True)


class Author(PayloadModel):
    role: str


class Content(PayloadModel):
    content_type: str
    parts: list[Any] = Field(default_factory=list)


class Metadata(PayloadModel):
    is_visually_hidden_from_conversation: bool = False
    is_redacted: bool = False
    is_complete: bool | None = None
    attachments: list[Any] = Field(default_factory=list)


class Message(PayloadModel):
    id: str
    author: Author
    content: Content
    metadata: Metadata = Field(default_factory=Metadata)
    channel: str | None = None
    status: str | None = None
    end_turn: bool | None = None


class Node(PayloadModel):
    id: str
    message: Message | None = None
    parent: str | None = None
    children: list[str] = Field(default_factory=list)


class Conversation(PayloadModel):
    title: str
    mapping: dict[str, Node]
    linear_conversation: list[Node]
    current_node: str
    is_public: bool


class TextMessage(PayloadModel):
    node_id: str
    role: Literal["user", "assistant"]
    channel: str | None
    content_type: str
    parts: list[str]
    status: str | None
    is_complete: bool | None


class Omission(PayloadModel):
    node_id: str
    reason: Literal["structural", "hidden", "role", "reasoning", "non_text", "empty_text", "redacted", "attachments"]
    content_type: str | None = None


class Report(PayloadModel):
    mapping_nodes: int
    linear_nodes: int
    extracted_messages: int
    text_characters: int
    content_types: dict[str, int]
    stream_closed: bool
    chain_reaches_root: bool
    chain_matches_linear: bool
    linear_matches_mapping: bool
    off_branch_nodes: int
    final_assistant_marked_complete: bool
    unfinished_message_ids: list[str]
    omissions: list[Omission]
    warnings: list[str]


class Extraction(PayloadModel):
    source_url: str
    title: str
    messages: list[TextMessage]
    report: Report


class StreamScripts(HTMLParser):
    """Collect only inline scripts; never evaluate them or fetch their imports."""

    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[str] = []
        self.chunks: list[str] | None = None

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Unlike the parent no-op, start collecting inline script text only."""
        if tag == "script" and "src" not in dict(attrs):
            self.chunks = []

    @override
    def handle_data(self, data: str) -> None:
        """Unlike the parent no-op, retain text only inside the current script."""
        if self.chunks is not None:
            self.chunks.append(data)

    @override
    def handle_endtag(self, tag: str) -> None:
        """Unlike the parent no-op, finish a collected script at its closing tag."""
        if tag == "script" and self.chunks is not None:
            self.scripts.append("".join(self.chunks).strip())
            self.chunks = None


class ReferenceTable:
    """Decode the observed stream's indexed containers, rejecting unknown encodings."""

    def __init__(self, values: list[Any]) -> None:
        self.values = values
        self.cache: dict[int, Any] = {}
        self.active: set[int] = set()

    def value(self, index: int) -> Any:
        """Return a table entry for a nonnegative integer reference, or reject it."""
        if type(index) is not int or not 0 <= index < len(self.values):
            raise ValueError("Invalid or out-of-range stream reference")
        return self.values[index]

    def fields(self, index: int) -> dict[str, int]:
        """Resolve object keys only, so callers can select a route before decoding.

        ``index`` points to an object whose keys are ``_N`` references to string
        table entries and whose values reference other entries. Return a mapping
        from resolved field names to still-encoded value references. This lets
        the caller avoid decoding unrelated bootstrap/promise data.
        """
        raw = self.value(index)
        if not isinstance(raw, dict):
            raise ValueError("Expected a reference-table object")
        result = {}
        for key, ref in raw.items():
            if not key.startswith("_") or not key[1:].isdigit():
                raise ValueError("Unsupported stream object-key encoding")
            name = self.value(int(key[1:]))
            if not isinstance(name, str) or name in result or type(ref) is not int:
                raise ValueError("Invalid or duplicate stream field")
            result[name] = ref
        return result

    def decode(self, index: int) -> Any:
        """Expand a selected entry into JSON-compatible containers and scalar data.

        ``index`` is a table reference, not a literal numeric value. Container
        integers reference entries; scalar numbers stored in the table remain
        numbers. The observed ``-5`` null sentinel becomes ``None``. Unknown
        negative references, tagged values (including deferred promises), cycles,
        and excessive nesting fail explicitly rather than producing partial text.
        """
        if type(index) is int and index == -5:
            return None
        if index in self.active:
            raise ValueError("Cyclic stream references")
        if index in self.cache:
            return self.cache[index]
        if len(self.active) >= MAX_REFERENCE_DEPTH:
            raise ValueError("Stream reference nesting exceeds the supported limit")
        raw = self.value(index)
        self.active.add(index)
        try:
            if isinstance(raw, dict):
                result = {key: self.decode(ref) for key, ref in self.fields(index).items()}
            elif isinstance(raw, list):
                if any(type(ref) is not int for ref in raw):
                    raise ValueError("Unsupported tagged/deferred stream value in the share route")
                result = [self.decode(ref) for ref in raw]
            else:
                result = raw
            self.cache[index] = result
            return result
        finally:
            self.active.remove(index)


def read_route(html: str) -> tuple[str, Conversation, bool]:
    """Find the share route in HTML without treating scripts as executable code.

    ``html`` is the full downloaded document. Return ``(source_url, conversation,
    stream_closed)``: the canonical URL uses the embedded share UUID; the validated
    conversation supplies the graph and ordered nodes; the boolean records an
    observed stream-close script, used as a transport-completeness check.

    The supported producer emits JSON-string arguments to ``enqueue``. Joining
    those strings produces a reference table followed by optional async records.
    Only ``loaderData -> routes/share.*`` is decoded. Its ``serverResponse`` must
    have ``type='data'`` and a ``data`` object with title, mapping, current_node,
    linear_conversation, and is_public. Async records outside that route are
    ignored; a deferred value inside the route is rejected by ReferenceTable.
    """
    parser = StreamScripts()
    parser.feed(html)
    chunks = []
    decoder = json.JSONDecoder()
    for script in parser.scripts:
        if script.startswith(STREAM_PREFIX):
            encoded = script[len(STREAM_PREFIX) :]
            chunk, end = decoder.raw_decode(encoded.lstrip())
            if not isinstance(chunk, str) or encoded.lstrip()[end:].strip() not in (");", ")"):
                raise ValueError("Unsupported enqueue argument; expected a JSON string literal")
            chunks.append(chunk)
    if not chunks:
        raise ValueError("No supported share stream found; this may be a login, challenge, or changed HTML page")
    values, _ = decoder.raw_decode("".join(chunks).lstrip())
    if not isinstance(values, list):
        raise ValueError("Expected an initial stream reference table")
    table = ReferenceTable(values)
    loader = table.fields(table.fields(0)["loaderData"])
    route_refs = [ref for key, ref in loader.items() if key.startswith("routes/share.")]
    if len(route_refs) != 1:
        raise ValueError("Expected exactly one ChatGPT share route")
    route = table.decode(route_refs[0])
    share_id = UUID(route["sharedConversationId"])
    response = route["serverResponse"]
    if response["type"] != "data":
        raise ValueError("Share route did not return conversation data")
    conversation = Conversation.model_validate(response["data"])
    return f"https://chatgpt.com/share/{share_id}", conversation, STREAM_CLOSE in parser.scripts


def extract(html: str) -> Extraction:
    """Return visible text and explicit coverage checks for one downloaded share.

    ``html`` must use the stream schema documented in ``read_route``. The result
    contains ``source_url``, ``title``, ``messages``, and ``report``. Each message
    preserves node ID, user/assistant role, channel, content type, string parts,
    status, and the producer's completion marker. Consumers read ``parts`` in
    order; parts are not stripped or joined, preserving code and Unicode exactly.

    The report compares the current node's parent chain with the linear list and
    compares list records with their graph counterparts. It records omissions
    (structural, hidden, role, reasoning, non_text, empty_text, redacted, attachments),
    counts, unfinished visible messages, and warnings. One node may have several
    omission records. Warnings and non-text omissions must be disclosed by callers;
    matching graph checks establish internal consistency, not completeness of an
    unseen private chat. Hidden/context and reasoning records never become text.
    """
    source_url, conversation, stream_closed = read_route(html)
    nodes = conversation.linear_conversation
    mapping = conversation.mapping
    chain = []
    seen = set()
    cursor = conversation.current_node
    while cursor is not None and cursor in mapping and cursor not in seen:
        seen.add(cursor)
        chain.append(cursor)
        cursor = mapping[cursor].parent
    chain.reverse()
    linear_ids = [node.id for node in nodes]
    reaches_root = cursor is None
    matches_chain = reaches_root and chain == linear_ids
    matches_mapping = len(set(linear_ids)) == len(nodes) and all(mapping.get(node.id) == node for node in nodes)
    warnings = []
    for passed, warning in [
        (stream_closed, "The HTML does not contain the expected stream-close marker."),
        (conversation.is_public, "The embedded payload is not marked public."),
        (reaches_root, "The current-node parent chain has a missing node or cycle."),
        (matches_chain, "The linear conversation differs from the current-node parent chain."),
        (matches_mapping, "The linear conversation has duplicate nodes or differs from the graph records."),
    ]:
        if not passed:
            warnings.append(warning)
    messages = []
    omissions = []
    content_types = Counter()
    for node in nodes:
        message = node.message
        if message is None:
            omissions.append(Omission(node_id=node.id, reason="structural"))
            continue
        content = message.content
        content_types[content.content_type] += 1
        reason = None
        if message.metadata.is_visually_hidden_from_conversation:
            reason = "hidden"
        elif message.author.role not in ("user", "assistant"):
            reason = "role"
        elif message.channel == "analysis" or content.content_type in ("thoughts", "reasoning_recap", "reasoning"):
            reason = "reasoning"
        elif message.metadata.is_redacted:
            reason = "redacted"
        elif content.content_type not in ("text", "multimodal_text"):
            reason = "non_text"
        if reason:
            omissions.append(Omission(node_id=node.id, reason=reason, content_type=content.content_type))
            continue
        parts = [part for part in content.parts if isinstance(part, str)]
        if len(parts) != len(content.parts):
            omissions.append(Omission(node_id=node.id, reason="non_text", content_type=content.content_type))
        if message.metadata.attachments:
            omissions.append(Omission(node_id=node.id, reason="attachments", content_type=content.content_type))
        if not any(parts):
            omissions.append(Omission(node_id=node.id, reason="empty_text", content_type=content.content_type))
            continue
        messages.append(
            TextMessage(
                node_id=node.id,
                role=message.author.role,
                channel=message.channel,
                content_type=content.content_type,
                parts=parts,
                status=message.status,
                is_complete=message.metadata.is_complete,
            )
        )
    final = nodes[-1].message if nodes else None
    final_complete = bool(final and final.author.role == "assistant" and final.status == "finished_successfully" and final.end_turn and final.metadata.is_complete is True)
    unfinished = [message.node_id for message in messages if message.status != "finished_successfully"]
    if not final_complete:
        warnings.append("The final node is not explicitly marked as a completed assistant answer.")
    if unfinished:
        warnings.append("Some extracted messages do not have finished_successfully status.")
    if any(item.reason in ("non_text", "attachments", "redacted") for item in omissions):
        warnings.append("Non-text or redacted material was omitted; this is not a full-fidelity transcript.")
    if not messages:
        warnings.append("No visible user/assistant text was found.")
    report = Report(
        mapping_nodes=len(mapping),
        linear_nodes=len(nodes),
        extracted_messages=len(messages),
        text_characters=sum(len(part) for message in messages for part in message.parts),
        content_types=dict(content_types),
        stream_closed=stream_closed,
        chain_reaches_root=reaches_root,
        chain_matches_linear=matches_chain,
        linear_matches_mapping=matches_mapping,
        off_branch_nodes=len(set(mapping) - seen),
        final_assistant_marked_complete=final_complete,
        unfinished_message_ids=unfinished,
        omissions=omissions,
        warnings=warnings,
    )
    return Extraction(source_url=source_url, title=conversation.title, messages=messages, report=report)


@click.command()
@click.option("report_only", "--report-only", is_flag=True, help="Print title, source URL, and coverage report without message text.")
def main(report_only: bool) -> None:
    """Read downloaded share HTML from stdin; write filtered JSON to stdout."""
    raw = sys.stdin.buffer.read(MAX_HTML_BYTES + 1)
    if len(raw) > MAX_HTML_BYTES:
        raise click.ClickException("HTML exceeds the 16 MiB input limit")
    try:
        result = extract(raw.decode("utf-8"))
    except (ValueError, KeyError, TypeError, RecursionError, ValidationError):
        # Validation errors can embed offending payload values, including hidden
        # context. Report the failure without printing any of those raw values.
        raise click.ClickException("Unsupported or malformed share payload. Inspect its structure locally; no transcript was emitted.") from None
    click.echo(result.model_dump_json(indent=2, exclude={"messages"} if report_only else None))


if __name__ == "__main__":
    main()
