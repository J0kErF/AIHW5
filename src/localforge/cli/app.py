"""localforge CLI entry point (docs/SPECIFICATION.md §6).

Hosts the lightweight commands (`version`, `baseline`, `pull`, `run`) and imports
the heavier command modules so they register on the shared Typer ``app``. Heavy
dependencies are imported lazily inside each command, so ``--help`` stays fast
and a missing optional extra never breaks the command surface.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.table import Table

from localforge import __version__
from localforge.cli._shared import DEFAULT_RESULTS, app, console, fail, metrics_table
from localforge.core.errors import AuthError, ConfigError
from localforge.core.logging import configure_logging
from localforge.core.types import Backend, Dtype

__all__ = ["app"]


@app.command()
def version() -> None:
    """Print the localforge version."""
    console.print(f"localforge {__version__}")


@app.command()
def baseline(
    results_dir: Path = typer.Option(DEFAULT_RESULTS, help="Where to write results."),
) -> None:
    """Record and print the idle RAM/VRAM baseline."""
    configure_logging(logging.WARNING)
    from localforge.profiling.baseline import collect_baseline, write_baseline

    record = collect_baseline()
    path = write_baseline(results_dir)
    table = Table(header_style="bold white on #01133f")
    table.add_column("metric")
    table.add_column("value")
    for key, value in record.items():
        table.add_row(key, str(value))
    console.print(table)
    console.print(f"[dim]written to {path}[/]")


@app.command()
def pull(model_id: str = typer.Argument(..., help="Hugging Face repo id.")) -> None:
    """Download a model from the Hugging Face Hub into the local cache."""
    configure_logging(logging.INFO)
    from localforge.config.settings import load_settings
    from localforge.models.acquire import pull_model

    try:
        info = pull_model(model_id, load_settings())
    except (AuthError, ConfigError) as exc:
        fail(exc)
    console.print(
        f"[green]✓[/] {info.model_id}  [{info.format}]  {info.size_mb:.0f} MB\n[dim]{info.path}[/]"
    )


@app.command()
def run(
    model: str = typer.Option(..., help="HF repo id or Ollama tag."),
    backend: Backend = typer.Option(Backend.TRANSFORMERS, help="Inference backend."),
    prompt: str = typer.Option("Explain virtual memory in one sentence.", help="Prompt."),
    max_new_tokens: int = typer.Option(64, min=1, max=4096),
    dtype: Dtype = typer.Option(Dtype.FP32, help="Precision (nf4/fp16 need CUDA)."),
) -> None:
    """Run a single inference and print its text + profiled metrics."""
    configure_logging(logging.WARNING)
    from localforge.backends.runner import run_spec
    from localforge.core.types import RunSpec

    spec = RunSpec(
        model_id=model,
        backend=backend,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        dtype=dtype,
    )
    result = run_spec(spec)
    console.print(metrics_table([result]))
    if result.backend_available:
        console.print(f"\n[bold]output:[/] {result.text.strip()}")
    else:
        console.print(f"\n[yellow]skipped:[/] {result.note}")


# Register the heavier commands (side-effect imports).
from localforge.cli import (  # noqa: E402,F401
    compare_cmd,
    econ_cmd,
    finetune_cmd,
    visualize_cmd,
)

if __name__ == "__main__":
    app()
