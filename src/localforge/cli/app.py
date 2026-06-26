"""localforge CLI entry point.

Commands are registered here. To keep `--help` fast and import-light, each
command's heavy dependencies (torch, backends, viz) are imported lazily inside
the command body rather than at module load. Command implementations land in
their own modules during Phase 4+ (see docs/TASKS.md T12, T16).
"""

from __future__ import annotations

import typer

from localforge import __version__

app = typer.Typer(
    name="localforge",
    help="Forge LLMs on the hardware you have: local inference, profiling, "
    "fine-tuning, and an OS-style paging visualizer.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the localforge version."""
    typer.echo(f"localforge {__version__}")


if __name__ == "__main__":
    app()
