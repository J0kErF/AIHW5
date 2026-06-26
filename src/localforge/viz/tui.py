"""Textual TUI for the paging visualizer (docs/SPECIFICATION.md §3.6).

A live dashboard that ingests the paging stream and shows the residency bar
across the memory hierarchy, running fault/hit/evict counters, and bytes
streamed. A ``--no-tui`` plain-text summary (:func:`plain_summary`) covers
SSH/CI where a full-screen TUI is unwanted.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Static

from localforge.core.types import PagingEvent
from localforge.paging.replay import summarize
from localforge.viz.state import HIERARCHY, PagingViewState


def plain_summary(events: list[PagingEvent], n_layers: int) -> str:
    s = summarize(events)
    state = PagingViewState()
    for e in events:
        state.ingest(e)
    bar = state.residency_bar(n_layers, width=n_layers)
    return (
        f"paging summary  ·  {s.n_events} events over {s.duration_ms:.0f} ms\n"
        f"  faults={s.n_faults}  hits={s.n_hits}  evicts={s.n_evicts}  "
        f"hit-rate={s.hit_rate:.0%}\n"
        f"  peak resident={s.peak_resident_layers}/{n_layers} blocks  "
        f"streamed={s.fault_bytes_mb:.0f} MB from disk\n"
        f"  hierarchy: {' -> '.join(HIERARCHY)}\n"
        f"  final residency [{bar}]"
    )


class PagingApp(App[None]):
    CSS = """
    #stats { padding: 1 2; height: auto; }
    """

    def __init__(self, events: list[PagingEvent], n_layers: int, interval: float = 0.04) -> None:
        super().__init__()
        self._events = events
        self._n = n_layers
        self._interval = interval
        self._i = 0
        self._state = PagingViewState()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(id="stats")
        yield Footer()

    def on_mount(self) -> None:
        self._render()
        self.set_interval(self._interval, self._tick)

    def _tick(self) -> None:
        if self._i >= len(self._events):
            return
        self._state.ingest(self._events[self._i])
        self._i += 1
        self._render()

    def _render(self) -> None:
        s = self._state
        bar = s.residency_bar(self._n, width=self._n)
        progress = f"{self._i}/{len(self._events)}"
        text = (
            f"[b]localforge paging visualizer[/]  ({progress} events)\n\n"
            f"faults [b]{s.faults}[/]   hits [b]{s.hits}[/]   evicts [b]{s.evicts}[/]   "
            f"resident [b]{s.resident_count}/{self._n}[/]   "
            f"streamed [b]{s.fault_mb:.0f} MB[/]   hit-rate [b]{s.hit_rate:.0%}[/]\n\n"
            f"residency [{bar}]\n\n"
            f"memory hierarchy:  {'  >  '.join(HIERARCHY)}"
        )
        self.query_one("#stats", Static).update(text)
