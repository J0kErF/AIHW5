"""Standalone HTML render of a paging trace (docs/SPECIFICATION.md §3.6).

Bundles the summary numbers, the memory-hierarchy framing, and the timeline PNG
into a single page so the paging visualization appears in reports without a live
terminal.
"""

from __future__ import annotations

from pathlib import Path

from localforge.paging.replay import PagingSummary
from localforge.viz.state import HIERARCHY

_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>localforge — paging visualizer</title>
<style>
 body {{ margin:0; background:#faf9f5; color:#01133f;
   font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
 .wrap {{ max-width:900px; margin:0 auto; padding:32px 24px; }}
 h1 {{ border-bottom:3px solid #cc5500; padding-bottom:10px; }}
 h1 small {{ color:#cc5500; }}
 .grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:18px 0; }}
 .card {{ background:#fff; border:1px solid #e4e1d8; border-radius:8px; padding:12px; }}
 .card .v {{ font-size:24px; font-weight:700; }}
 .card .k {{ color:#666; font-size:12px; }}
 .hier {{ display:flex; gap:6px; margin:14px 0; }}
 .hier div {{ flex:1; text-align:center; padding:8px 4px; border-radius:6px; font-size:12px; color:#fff; }}
 img {{ max-width:100%; border:1px solid #e4e1d8; background:#fff; padding:8px; }}
 p.note {{ color:#555; font-size:14px; }}
</style></head><body><div class="wrap">
 <h1>localforge <small>paging visualizer</small></h1>
 <p class="note">AirLLM streams a model one transformer block at a time, loading each block
   from disk (SafeTensors + mmap) right before it runs, then evicting it — exactly how an
   operating system pages memory between RAM and disk. Below: the measured/modeled trace.</p>
 <div class="grid">
  <div class="card"><div class="v">{faults}</div><div class="k">page faults</div></div>
  <div class="card"><div class="v">{evicts}</div><div class="k">evictions</div></div>
  <div class="card"><div class="v">{peak}</div><div class="k">peak resident blocks</div></div>
  <div class="card"><div class="v">{mb:.0f} MB</div><div class="k">streamed from disk</div></div>
 </div>
 <div class="hier">{hierarchy}</div>
 {image}
 <p class="note">Hit rate {hit:.0%} over {events} events spanning {dur:.0f} ms.
   A bounded resident set ({peak} blocks) is what lets a model far larger than RAM run locally.</p>
</div></body></html>
"""


def render_paging_html(
    summary: PagingSummary,
    out: Path,
    png_name: str | None = None,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    shades = ["#01133f", "#23386b", "#4a6aa5", "#7e57c2", "#cc5500"]
    hierarchy = "".join(
        f'<div style="background:{shades[i % len(shades)]}">{level}</div>'
        for i, level in enumerate(HIERARCHY)
    )
    image = f'<img src="{png_name}" alt="paging timeline">' if png_name else ""
    html = _TEMPLATE.format(
        faults=summary.n_faults,
        evicts=summary.n_evicts,
        peak=summary.peak_resident_layers,
        mb=summary.fault_bytes_mb,
        hierarchy=hierarchy,
        image=image,
        hit=summary.hit_rate,
        events=summary.n_events,
        dur=summary.duration_ms,
    )
    out.write_text(html, encoding="utf-8")
    return out
