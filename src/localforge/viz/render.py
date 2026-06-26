"""Export paging artifacts (docs/SPECIFICATION.md §3.6).

Writes the stream JSONL, the timeline PNG, and the standalone HTML so the paging
visualization appears in reports and survives without a live terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from localforge.core.types import PagingEvent
from localforge.paging.events import write_stream
from localforge.paging.replay import summarize
from localforge.viz.html import render_paging_html
from localforge.viz.static import render_timeline_png


@dataclass
class PagingArtifacts:
    jsonl: Path
    png: Path
    html: Path


def export_paging_artifacts(
    events: list[PagingEvent],
    out_dir: Path,
    name: str = "paging",
    title: str = "AirLLM paging timeline",
) -> PagingArtifacts:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = write_stream(events, out_dir / f"{name}.jsonl")
    png = render_timeline_png(events, out_dir / f"{name}.png", title=title)
    summary = summarize(events)
    html = render_paging_html(summary, out_dir / f"{name}.html", png_name=png.name)
    return PagingArtifacts(jsonl=jsonl, png=png, html=html)
