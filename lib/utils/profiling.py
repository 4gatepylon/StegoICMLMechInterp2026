"""Append component timings without conflating concurrent work and wall time."""

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from threading import Lock
from time import perf_counter
from typing import Iterator, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class TimingRecord(BaseModel):
    """One inclusive duration; overlapping/nested records must never be added.

    ``invocation_id`` separates preparation, generation, grading, and resumes.
    ``component`` names the measured operation; ``model`` and ``request_id`` are
    nullable attribution keys. ``elapsed_s`` uses a monotonic local clock or an
    SDK/remote-reported duration (``source`` distinguishes them). ``started_at``
    is the local observation time in UTC, not a remote clock synchronization.
    ``items`` counts completed items only on success; failed attempts count zero.
    ``status`` distinguishes success, ordinary errors, and caller interruption.
    """

    model_config = ConfigDict(extra="forbid")
    invocation_id: str
    component: str
    model: str | None = None
    request_id: str | None = None
    started_at: str
    elapsed_s: float = Field(ge=0, allow_inf_nan=False)
    items: int = Field(default=0, ge=0)
    status: Literal["ok", "error", "interrupted"] = "ok"
    source: Literal["local", "sdk", "remote"] = "local"


_active: ContextVar[tuple["Profiler", str | None, str | None] | None] = ContextVar("component_profiler", default=None)


class Profiler:
    """Thread-safe append-only recorder scoped to one operation invocation.

    ``path`` is an already-resolved artifact path whose parent must exist. Each
    flushed JSONL row follows TimingRecord; no prompts, secrets, or code are saved.
    Call ``bind`` inside each worker because context variables do not propagate
    automatically into ThreadPoolExecutor threads. Existing incomplete JSONL is
    rejected before appending. Separate processes must not share a run directory.
    """

    def __init__(self, path: Path):
        if path.exists():
            read_timings(path)
        self.path = path
        self.invocation_id = uuid4().hex
        self.lock = Lock()

    @contextmanager
    def bind(self, *, model: str | None = None, request_id: str | None = None) -> Iterator[None]:
        """Attribute nested measures to this recorder/model/request until exit."""
        token = _active.set((self, model, request_id))
        try:
            yield
        finally:
            _active.reset(token)

    def write(self, **fields) -> None:
        """Validate TimingRecord fields and flush one row under the writer lock."""
        row = TimingRecord(invocation_id=self.invocation_id, **fields)
        with self.lock, self.path.open("a") as output:
            output.write(row.model_dump_json() + "\n")
            output.flush()


@contextmanager
def measure(component: str, *, items: int = 0, start_s: float | None = None) -> Iterator[None]:
    """Measure an inclusive operation in the active profile, or do nothing.

    ``component`` is a stable aggregation name; ``items`` counts work completed
    only if the body returns normally. Exceptions propagate after a timing row is
    flushed, including KeyboardInterrupt/SystemExit. Nested durations overlap.
    ``start_s`` optionally includes work begun before this context was bound,
    using a previously captured perf_counter value. The UTC time remains the
    observation time. No binding preserves existing unprofiled caller behavior.
    """
    active = _active.get()
    if active is None:
        yield
        return
    profiler, model, request_id = active
    started_at = datetime.now(timezone.utc).isoformat()
    started = perf_counter() if start_s is None else start_s
    status = "ok"
    try:
        yield
    except BaseException as error:
        status = "error" if isinstance(error, Exception) else "interrupted"
        raise
    finally:
        elapsed = perf_counter() - started
        profiler.write(component=component, model=model, request_id=request_id, started_at=started_at, elapsed_s=elapsed, items=items if status == "ok" else 0, status=status)


def report_duration(component: str, elapsed_s: float, *, source: Literal["local", "sdk", "remote"], items: int = 0) -> None:
    """Record a previously measured duration under the active attribution keys.

    ``source`` distinguishes locally captured intervals from SDK/remote reports.
    All are inclusive observations, not additional nonoverlapping wall time.
    Missing measurements should not call this function; they are never zero-filled.
    """
    if (active := _active.get()) is not None:
        profiler, model, request_id = active
        profiler.write(component=component, model=model, request_id=request_id, started_at=datetime.now(timezone.utc).isoformat(), elapsed_s=elapsed_s, source=source, items=items)


def read_timings(path: Path) -> list[TimingRecord]:
    """Read validated timing rows; absent legacy profiles yield an empty list."""
    if not path.exists():
        return []
    text = path.read_text()
    if text and not text.endswith("\n"):
        raise ValueError(f"Incomplete profiling record in {path}; preserve and repair before resuming")
    return [TimingRecord.model_validate_json(line) for line in text.splitlines()]


def summarize_timings(path: Path) -> list[dict]:
    """Aggregate inclusive timings by invocation/component/model/source.

    Returns dictionaries with those four grouping keys, count, errors, interrupted,
    total_s, mean_s, median_s, p90_s (linear interpolation), max_s, completed_items,
    and items_per_s. Durations include failed attempts; throughput uses successful
    items divided by all observed duration. Only a stage's local total measures
    wall-clock throughput; summed request/worker times measure overlapping work.
    Empty legacy profiles return []; unknown measurements are never invented.
    """
    groups = defaultdict(list)
    for row in read_timings(path):
        groups[(row.invocation_id, row.component, row.model, row.source)].append(row)
    summaries = []
    for (invocation_id, component, model, source), rows in groups.items():
        seconds = sorted(row.elapsed_s for row in rows)
        rank = (len(seconds) - 1) * 0.9
        lower = int(rank)
        p90 = seconds[lower] + (seconds[min(lower + 1, len(seconds) - 1)] - seconds[lower]) * (rank - lower)
        total = sum(seconds)
        items = sum(row.items for row in rows)
        summaries.append(
            dict(
                invocation_id=invocation_id,
                component=component,
                model=model,
                source=source,
                count=len(rows),
                errors=sum(row.status == "error" for row in rows),
                interrupted=sum(row.status == "interrupted" for row in rows),
                total_s=total,
                mean_s=mean(seconds),
                median_s=median(seconds),
                p90_s=p90,
                max_s=max(seconds),
                completed_items=items,
                items_per_s=items / total if total and items else None,
            )
        )
    return summaries
