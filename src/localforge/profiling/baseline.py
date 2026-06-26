"""Idle hardware baseline (docs/SPECIFICATION.md §3.3.1).

Records idle RAM/VRAM before any model loads, so the comparison report can show
each run's cost *above* the host's resting state.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

from localforge.profiling.probes import (
    cuda_current_mb,
    current_rss_mb,
    system_available_mb,
)


def collect_baseline() -> dict[str, Any]:
    """Sample the idle baseline. Pure; safe on CPU-only hosts (vram -> None)."""
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "idle_process_rss_mb": round(current_rss_mb(), 2),
        "system_available_mb": round(system_available_mb(), 2),
        "idle_vram_mb": cuda_current_mb(),
    }


def write_baseline(results_dir: Path) -> Path:
    """Collect the baseline and persist it to ``<results_dir>/baseline/<host>.json``."""
    record = collect_baseline()
    out_dir = results_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{record['host'] or 'localhost'}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path
