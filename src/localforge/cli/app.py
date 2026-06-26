"""localforge CLI (docs/SPECIFICATION.md §6).

Commands: ``version``, ``baseline``, ``pull``, ``run``, ``compare``. Heavy
dependencies (torch, backends, matplotlib) are imported lazily inside each
command so ``--help`` stays fast and a missing optional extra never breaks the
command surface. Typed LocalforgeErrors map to clean exit codes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from localforge import __version__
from localforge.core.errors import AuthError, ConfigError, LocalforgeError
from localforge.core.logging import configure_logging
from localforge.core.types import Backend, Dtype, RunResult

app = typer.Typer(
    name="localforge",
    help="Forge LLMs on the hardware you have: local inference, profiling, "
    "fine-tuning, and an OS-style paging visualizer.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

_DEFAULT_RESULTS = Path("results")


def _fail(exc: LocalforgeError) -> None:
    """Map a typed error to a clean message + exit code."""
    code = 2 if isinstance(exc, ConfigError) else 1
    console.print(f"[bold red]error:[/] {exc}")
    raise typer.Exit(code) from exc


def _metrics_table(results: list[RunResult]) -> Table:
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
        status = "ok" if r.backend_available else "skipped"
        style = "" if r.backend_available else "dim italic"
        table.add_row(
            r.backend.value,
            r.model_id,
            status,
            num(r.load_s if r.backend_available else None),
            num(r.prefill_ms if r.backend_available else None),
            num(r.decode_tok_s),
            num(r.peak_ram_mb if r.backend_available else None),
            num(r.peak_vram_mb),
            (r.note or ""),
            style=style,
        )
    return table


@app.command()
def version() -> None:
    """Print the localforge version."""
    console.print(f"localforge {__version__}")


@app.command()
def baseline(
    results_dir: Path = typer.Option(_DEFAULT_RESULTS, help="Where to write results."),
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
        _fail(exc)
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
    console.print(_metrics_table([result]))
    if result.backend_available:
        console.print(f"\n[bold]output:[/] {result.text.strip()}")
    else:
        console.print(f"\n[yellow]skipped:[/] {result.note}")


@app.command()
def compare(
    suite: Path = typer.Option(..., exists=True, help="Path to a suite YAML."),
    backend: Backend | None = typer.Option(None, help="Only run this backend."),
    results_dir: Path = typer.Option(_DEFAULT_RESULTS, help="Where to write results."),
) -> None:
    """Run a comparison suite and write a Markdown/JSON/HTML report."""
    configure_logging(logging.INFO)
    from localforge.backends.runner import run_spec
    from localforge.config.suite import expand_suite, load_suite
    from localforge.reporting.report import write_report
    from localforge.reporting.store import ResultStore

    try:
        doc = load_suite(suite)
        specs = expand_suite(doc)
    except ConfigError as exc:
        _fail(exc)

    if backend is not None:
        specs = [s for s in specs if s.backend is backend]
    if not specs:
        _fail(ConfigError("no runs match the requested backend filter"))

    results: list[RunResult] = []
    for spec in specs:
        console.print(
            f"[dim]running[/] {spec.backend.value} · {spec.model_id} · {spec.dtype.value}"
        )
        results.append(run_spec(spec))

    store = ResultStore(results_dir)
    store.save_suite(doc.suite_id, results)
    suite_run = store.load_suite(doc.suite_id)
    assert suite_run is not None
    paths = write_report(doc.suite_id, results, suite_run, results_dir / "reports")

    console.print(_metrics_table(results))
    console.print(f"\n[green]report:[/] {paths.report_html}")


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
        _fail(ConfigError(f"unknown method {method!r}; use lora|qlora|olora"))

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


@app.command()
def visualize(
    replay: Path | None = typer.Option(None, exists=True, help="Replay a recorded JSONL stream."),
    model: str | None = typer.Option(None, help="Capture a real per-block trace from this model."),
    layers: int = typer.Option(32, help="Block count for the synthetic trace."),
    max_new_tokens: int = typer.Option(16, min=1, help="Tokens for live capture."),
    no_tui: bool = typer.Option(
        False, "--no-tui", help="Print a plain summary instead of the TUI."
    ),
    results_dir: Path = typer.Option(_DEFAULT_RESULTS, help="Where to write artifacts."),
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


if __name__ == "__main__":
    app()
