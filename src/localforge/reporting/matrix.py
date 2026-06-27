"""Comparison matrix assembly (docs/SPECIFICATION.md §3.4).

Turns a list of RunResults into rows for the report. Unavailable backends are
kept as explicit 'skipped (reason)' rows rather than dropped, so the comparison
is honest about what did and did not run.
"""

from __future__ import annotations

import json
from typing import Any

from localforge.core.types import RunResult

COLUMNS = [
    ("backend", "Backend"),
    ("model_id", "Model"),
    ("status", "Status"),
    ("load_s", "Load (s)"),
    ("ttft_ms", "TTFT (ms)"),
    ("tpot_ms", "TPOT (ms)"),
    ("decode_tok_s", "Throughput (tok/s)"),
    ("peak_ram_mb", "Peak RAM (MB)"),
    ("peak_vram_mb", "Peak VRAM (MB)"),
    ("note", "Note"),
]


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def build_rows(results: list[RunResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in results:
        if r.backend_available:
            rows.append(
                {
                    "backend": r.backend.value,
                    "model_id": r.model_id,
                    "status": "ok",
                    "load_s": r.load_s,
                    "ttft_ms": r.ttft_ms,
                    "tpot_ms": r.tpot_ms,
                    "decode_tok_s": r.decode_tok_s,
                    "peak_ram_mb": r.peak_ram_mb,
                    "peak_vram_mb": r.peak_vram_mb,
                    "note": r.note,
                }
            )
        else:
            rows.append(
                {
                    "backend": r.backend.value,
                    "model_id": r.model_id,
                    "status": "skipped",
                    "load_s": None,
                    "ttft_ms": None,
                    "tpot_ms": None,
                    "decode_tok_s": None,
                    "peak_ram_mb": None,
                    "peak_vram_mb": None,
                    "note": r.note,
                }
            )
    return rows


def to_markdown(rows: list[dict[str, Any]]) -> str:
    header = "| " + " | ".join(label for _, label in COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key, _ in COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def to_json(rows: list[dict[str, Any]]) -> str:
    return json.dumps(rows, indent=2)
