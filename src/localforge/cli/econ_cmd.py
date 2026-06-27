"""The `econ` command — API-vs-OnPrem breakeven analysis (spec §5.5)."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.table import Table

from localforge.cli._shared import app, console
from localforge.core.logging import configure_logging


@app.command()
def econ(
    api_usd_per_1m: float = typer.Option(0.60, help="API price per 1M tokens (USD)."),
    gpu_capex: float = typer.Option(1600.0, help="One-time GPU cost (USD)."),
    throughput_tok_s: float = typer.Option(40.0, help="OnPrem served tokens/sec."),
    figures_dir: Path = typer.Option(Path("figures"), help="Where to write the figure."),
) -> None:
    """Compute the API-vs-OnPrem breakeven and render a cost-vs-volume figure."""
    configure_logging(logging.WARNING)
    from localforge.econ.breakeven import CostAssumptions, analyze
    from localforge.econ.figure import render_breakeven_png

    assumptions = CostAssumptions(
        api_usd_per_1m_tokens=api_usd_per_1m,
        gpu_capex_usd=gpu_capex,
        onprem_throughput_tok_s=throughput_tok_s,
    )
    b = analyze(assumptions)
    png = render_breakeven_png(assumptions, figures_dir / "breakeven.png")

    table = Table(header_style="bold white on #01133f")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("API $/token", f"${b.api_per_tok:.2e}")
    table.add_row("OnPrem OPEX $/token", f"${b.onprem_opex_per_tok:.2e}")
    table.add_row("Cloud GPU $/token", f"${b.cloud_per_tok:.2e}")
    table.add_row("OnPrem CAPEX", f"${b.onprem_capex:,.0f}")
    if b.breakeven_millions is not None:
        table.add_row("Breakeven (OnPrem < API)", f"{b.breakeven_millions:,.0f}M tokens")
    else:
        table.add_row("Breakeven", "never (API OPEX already lower)")
    console.print(table)
    console.print(f"[green]figure:[/] {png}")
    console.print("[dim]Prices are illustrative 2026 assumptions; see reports/REPORT.md §econ.[/]")
