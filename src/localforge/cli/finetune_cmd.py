"""The `finetune` command (docs/SPECIFICATION.md §3.5)."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.table import Table

from localforge.cli._shared import app, console, fail
from localforge.core.errors import ConfigError
from localforge.core.logging import configure_logging


@app.command()
def finetune(
    model: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct", help="Base HF model."),
    method: str = typer.Option("lora", help="lora | qlora | olora."),
    data: Path = typer.Option(
        Path("data/finetune/tiny_sft.jsonl"), exists=True, help="JSONL SFT dataset."
    ),
    steps: int = typer.Option(12, min=1, help="Training steps."),
    out: Path | None = typer.Option(None, help="Adapter output dir."),
) -> None:
    """Train a LoRA/QLoRA/OLoRA adapter and show before/after generations."""
    configure_logging(logging.INFO)
    from localforge.config.settings import load_settings
    from localforge.finetune.adapters import FineTuneMethod
    from localforge.finetune.trainer import train_adapter

    try:
        ft_method = FineTuneMethod(method.lower())
    except ValueError:
        fail(ConfigError(f"unknown method {method!r}; use lora|qlora|olora"))

    result = train_adapter(model, ft_method, data, load_settings(), steps=steps, out_dir=out)
    table = Table(header_style="bold white on #01133f")
    table.add_column("field")
    table.add_column("value")
    table.add_row("method", result.method.value)
    table.add_row("trainable params", f"{result.trainable_params:,} ({result.trainable_pct:.3f}%)")
    table.add_row("total params", f"{result.total_params:,}")
    table.add_row("final loss", f"{result.final_loss:.4f}")
    table.add_row("adapter", result.adapter_path)
    if result.note:
        table.add_row("note", result.note)
    console.print(table)
    console.print(f"\n[bold]before:[/] {result.before}")
    console.print(f"[bold]after :[/] {result.after}")
