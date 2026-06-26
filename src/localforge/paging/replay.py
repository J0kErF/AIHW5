"""Replay and summarize paging streams (docs/SPECIFICATION.md §3.6).

Replay feeds a recorded stream to the tracer's sinks so the visualizer works
with no AirLLM installed. The summary aggregates a stream into the headline
numbers (faults, evicts, bytes moved, peak residency) used in the report and the
predicted-vs-measured cross-check (RE_AIRLLM.md, T15a).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from localforge.core.types import PageAction, PagingEvent
from localforge.paging.events import read_stream
from localforge.paging.tracer import PagingTracer


def load_and_replay(path: Path, tracer: PagingTracer) -> list[PagingEvent]:
    events = read_stream(path)
    tracer.replay(events)
    return events


@dataclass
class PagingSummary:
    n_events: int
    n_faults: int
    n_hits: int
    n_evicts: int
    distinct_layers: int
    fault_bytes: int
    peak_resident_layers: int
    duration_ms: float

    @property
    def hit_rate(self) -> float:
        lookups = self.n_faults + self.n_hits
        return self.n_hits / lookups if lookups else 0.0

    @property
    def fault_bytes_mb(self) -> float:
        return self.fault_bytes / (1024 * 1024)


def summarize(events: list[PagingEvent]) -> PagingSummary:
    n_faults = n_hits = n_evicts = 0
    fault_bytes = 0
    resident: set[int] = set()
    peak_resident = 0
    layers: set[int] = set()

    for ev in events:
        layers.add(ev.layer)
        if ev.action is PageAction.FAULT:
            n_faults += 1
            fault_bytes += ev.bytes
            resident.add(ev.layer)
            peak_resident = max(peak_resident, len(resident))
        elif ev.action is PageAction.HIT:
            n_hits += 1
        elif ev.action is PageAction.EVICT:
            n_evicts += 1
            resident.discard(ev.layer)

    duration = (events[-1].t_ms - events[0].t_ms) if events else 0.0
    return PagingSummary(
        n_events=len(events),
        n_faults=n_faults,
        n_hits=n_hits,
        n_evicts=n_evicts,
        distinct_layers=len(layers),
        fault_bytes=fault_bytes,
        peak_resident_layers=peak_resident,
        duration_ms=duration,
    )
