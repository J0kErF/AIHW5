"""Static PNG render of a paging trace (docs/SPECIFICATION.md §3.6).

Two panels tell the AirLLM paging story for the report: (top) per-block events
over time coloured by action (fault/hit/evict), and (bottom) the resident block
count over time — the bounded working set that lets a huge model run in little
RAM.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from localforge.core.types import PageAction, PagingEvent  # noqa: E402

_COLORS = {
    PageAction.FAULT: "#cc5500",
    PageAction.HIT: "#2e8b57",
    PageAction.EVICT: "#888888",
}


def render_timeline_png(
    events: list[PagingEvent],
    out: Path,
    title: str = "AirLLM paging timeline",
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for action, color in _COLORS.items():
        xs = [e.t_ms for e in events if e.action is action]
        ys = [e.layer for e in events if e.action is action]
        if xs:
            ax_top.scatter(xs, ys, s=18, c=color, label=action.value, alpha=0.8)
    ax_top.set_ylabel("Block index")
    ax_top.set_title(title)
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.grid(True, linestyle=":", alpha=0.4)

    # Resident-block count over time (distinct blocks currently in RAM).
    resident: set[int] = set()
    ts: list[float] = []
    counts: list[int] = []
    for e in events:
        if e.action is PageAction.FAULT:
            resident.add(e.layer)
        elif e.action is PageAction.EVICT:
            resident.discard(e.layer)
        ts.append(e.t_ms)
        counts.append(len(resident))
    ax_bot.step(ts, counts, where="post", color="#01133f")
    ax_bot.fill_between(ts, counts, step="post", alpha=0.15, color="#01133f")
    ax_bot.set_ylabel("Resident blocks")
    ax_bot.set_xlabel("Time (ms)")
    ax_bot.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
