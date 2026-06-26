"""Incremental view state for the paging visualizer (docs/SPECIFICATION.md §3.6).

Separated from the Textual UI so the display logic — residency set, fault/hit/
evict counters, bytes streamed, memory-hierarchy mapping — is plain and unit
testable. The TUI feeds events in via :meth:`ingest` and redraws from this state.
"""

from __future__ import annotations

from localforge.core.types import PageAction, PagingEvent

# Memory hierarchy levels, coldest (disk) to hottest (registers). AirLLM streams
# blocks up from disk through the OS page cache (mmap) into RAM for compute.
HIERARCHY = ["NVMe/SSD", "Page cache (mmap)", "RAM", "Cache", "Registers"]


class PagingViewState:
    def __init__(self) -> None:
        self.resident: set[int] = set()
        self.faults = 0
        self.hits = 0
        self.evicts = 0
        self.fault_bytes = 0
        self.last_event: PagingEvent | None = None

    def ingest(self, event: PagingEvent) -> None:
        self.last_event = event
        if event.action is PageAction.FAULT:
            self.faults += 1
            self.fault_bytes += event.bytes
            self.resident.add(event.layer)
        elif event.action is PageAction.HIT:
            self.hits += 1
        elif event.action is PageAction.EVICT:
            self.evicts += 1
            self.resident.discard(event.layer)

    @property
    def resident_count(self) -> int:
        return len(self.resident)

    @property
    def fault_mb(self) -> float:
        return self.fault_bytes / (1024 * 1024)

    @property
    def hit_rate(self) -> float:
        lookups = self.faults + self.hits
        return self.hits / lookups if lookups else 0.0

    def residency_bar(self, n_layers: int, width: int = 32) -> str:
        """ASCII bar of which blocks are resident (#) vs paged out (.).

        Deliberately ASCII so it renders in any terminal/CI/Windows console.
        """
        if n_layers <= 0:
            return ""
        bar = "".join("#" if layer in self.resident else "." for layer in range(n_layers))
        return bar[:width] if width and len(bar) > width else bar
