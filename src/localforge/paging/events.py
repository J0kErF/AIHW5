"""Paging event sinks and stream persistence (docs/SPECIFICATION.md §3.6).

A sink is any ``Callable[[PagingEvent], None]``. The tracer (Observer) fans each
event out to its sinks: a live TUI widget, a JSONL writer, and — in tests — an
in-memory list. Streams persist as JSONL so a run can be replayed/visualized
later without re-running AirLLM.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from localforge.core.types import PagingEvent

Sink = Callable[[PagingEvent], None]


class MemorySink:
    """Collects events in a list (used by tests and the static renderer)."""

    def __init__(self) -> None:
        self.events: list[PagingEvent] = []

    def __call__(self, event: PagingEvent) -> None:
        self.events.append(event)


class JsonlSink:
    """Appends each event as one JSON line to a file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def __call__(self, event: PagingEvent) -> None:
        self._fh.write(event.model_dump_json() + "\n")

    def close(self) -> None:
        self._fh.close()


def write_stream(events: list[PagingEvent], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(e.model_dump_json() for e in events) + "\n", encoding="utf-8")
    return path


def read_stream(path: Path) -> list[PagingEvent]:
    events: list[PagingEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(PagingEvent.model_validate_json(line))
    return events
