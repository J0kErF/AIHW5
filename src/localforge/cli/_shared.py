"""Shared CLI scaffolding (docs/SPECIFICATION.md §6).

The Typer ``app`` and small helpers live here so command modules can register on
it without import cycles, and so each CLI file stays focused and small.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from localforge.core.errors import ConfigError, LocalforgeError
from localforge.core.types import RunResult

app = typer.Typer(
    name="localforge",
    help="Forge LLMs on the hardware you have: local inference, profiling, "
    "fine-tuning, and an OS-style paging visualizer.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

DEFAULT_RESULTS = Path("results")


def fail(exc: LocalforgeError) -> None:
    """Map a typed error to a clean message + exit code (Config -> 2, else 1)."""
    code = 2 if isinstance(exc, ConfigError) else 1
    console.print(f"[bold red]error:[/] {exc}")
    raise typer.Exit(code) from exc


def metrics_table(results: list[RunResult]) -> Table:
    """Render a Rich table of profiled run metrics, including skipped rows."""
    table = Table(show_lines=False, header_style="bold white on #01133f")
    for col in (
        "backend",
        "model",
        "status",
        "load s",
        "prefill ms",
        "decode tok/s",
        "peak RAM MB",
        "peak VRAM MB",
        "note",
    ):
        table.add_column(col)

    def num(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1f}"

    for r in results:
        ok = r.backend_available
        table.add_row(
            r.backend.value,
            r.model_id,
            "ok" if ok else "skipped",
            num(r.load_s if ok else None),
            num(r.prefill_ms if ok else None),
            num(r.decode_tok_s),
            num(r.peak_ram_mb if ok else None),
            num(r.peak_vram_mb),
            (r.note or ""),
            style="" if ok else "dim italic",
        )
    return table
