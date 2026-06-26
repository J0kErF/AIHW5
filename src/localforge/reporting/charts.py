"""Static PNG charts for the comparison report (docs/SPECIFICATION.md §3.4).

Uses matplotlib's non-interactive Agg backend so charts render headless (CI, no
display). Only successful runs are plotted; charts are skipped gracefully when
there is nothing to show.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from localforge.core.types import RunResult  # noqa: E402


def _ok(results: list[RunResult]) -> list[RunResult]:
    return [r for r in results if r.backend_available]


def _label(r: RunResult) -> str:
    return f"{r.backend.value}\n{r.model_id.split('/')[-1]}"


def _bar(results: list[RunResult], attr: str, title: str, ylabel: str, out: Path) -> Path | None:
    points: list[tuple[str, float]] = []
    for r in _ok(results):
        value = getattr(r, attr)
        if value is not None:
            points.append((_label(r), float(value)))
    if not points:
        return None

    labels = [label for label, _ in points]
    values = [value for _, value in points]
    fig, ax = plt.subplots(figsize=(max(4.0, len(labels) * 1.6), 4.0))
    ax.bar(labels, values, color="#cc5500")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def render_charts(results: list[RunResult], out_dir: Path) -> list[Path]:
    """Render the standard comparison charts; return the files actually written."""
    specs = [
        ("peak_ram_mb", "Peak RAM by backend", "Peak RAM (MB)", "peak_ram.png"),
        ("decode_tok_s", "Decode throughput by backend", "tokens/sec", "decode_tok_s.png"),
        ("load_s", "Model load time by backend", "seconds", "load_s.png"),
    ]
    written: list[Path] = []
    for attr, title, ylabel, fname in specs:
        path = _bar(results, attr, title, ylabel, out_dir / fname)
        if path is not None:
            written.append(path)
    return written
