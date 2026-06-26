"""PagingTracer: the Observer that publishes paging events (docs/IMPLEMENTATION.md §2.3).

Capture is decoupled from rendering: the AirLLM hook (or a replay/synthetic
source) calls :meth:`emit`, and every subscribed sink — TUI widget, JSONL
writer, in-memory list — receives the event. Timestamps are milliseconds since
:meth:`start`.
"""

from __future__ import annotations

from time import perf_counter

from localforge.core.types import PageAction, PageSource, PagingEvent
from localforge.paging.events import Sink


class PagingTracer:
    def __init__(self) -> None:
        self._sinks: list[Sink] = []
        self._t0: float | None = None

    def subscribe(self, sink: Sink) -> PagingTracer:
        self._sinks.append(sink)
        return self

    def start(self) -> None:
        self._t0 = perf_counter()

    def emit(
        self,
        layer: int,
        action: PageAction,
        n_bytes: int,
        source: PageSource,
        *,
        t_ms: float | None = None,
    ) -> PagingEvent:
        if t_ms is None:
            t_ms = (perf_counter() - self._t0) * 1000.0 if self._t0 is not None else 0.0
        event = PagingEvent(layer=layer, action=action, bytes=n_bytes, source=source, t_ms=t_ms)
        for sink in self._sinks:
            sink(event)
        return event

    def replay(self, events: list[PagingEvent]) -> None:
        """Re-emit a recorded stream to all sinks (timestamps preserved)."""
        for event in events:
            for sink in self._sinks:
                sink(event)
