"""The `visualize` command (docs/SPECIFICATION.md §3.6)."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from localforge.cli._shared import DEFAULT_RESULTS, app, console
from localforge.core.logging import configure_logging


@app.command()
def visualize(
    replay: Path | None = typer.Option(None, exists=True, help="Replay a recorded JSONL stream."),
    model: str | None = typer.Option(None, help="Capture a real per-block trace from this model."),
    layers: int = typer.Option(32, help="Block count for the synthetic trace."),
    max_new_tokens: int = typer.Option(16, min=1, help="Tokens for live capture."),
    no_tui: bool = typer.Option(
        False, "--no-tui", help="Print a plain summary instead of the TUI."
    ),
    results_dir: Path = typer.Option(DEFAULT_RESULTS, help="Where to write artifacts."),
) -> None:
    """Visualize AirLLM-style paging: live capture, replay, or synthetic model."""
    configure_logging(logging.WARNING)
    from localforge.paging.airllm_hook import synthesize_airllm_trace
    from localforge.paging.events import read_stream
    from localforge.viz.render import export_paging_artifacts
    from localforge.viz.tui import PagingApp, plain_summary

    if replay is not None:
        events = read_stream(replay)
        title = f"Paging trace (replay: {replay.name})"
    elif model is not None:
        from localforge.viz.capture import capture_transformers_trace

        console.print(f"[dim]capturing per-block trace from {model}...[/]")
        events = capture_transformers_trace(
            model, "Explain virtual memory.", max_new_tokens=max_new_tokens
        )
        title = f"Per-block execution trace ({model})"
    else:
        events = synthesize_airllm_trace(n_layers=layers)
        title = "Synthetic AirLLM paging model"

    n_layers = (max(e.layer for e in events) + 1) if events else layers
    artifacts = export_paging_artifacts(events, results_dir / "paging", title=title)
    console.print(f"[green]artifacts:[/] {artifacts.png}  ·  {artifacts.html}")

    if no_tui:
        console.print(plain_summary(events, n_layers))
    else:
        PagingApp(events, n_layers).run()
