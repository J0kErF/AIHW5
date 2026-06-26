"""The `compare` command (docs/SPECIFICATION.md §3.4)."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from localforge.cli._shared import DEFAULT_RESULTS, app, console, fail, metrics_table
from localforge.core.errors import ConfigError
from localforge.core.logging import configure_logging
from localforge.core.types import Backend, RunResult


@app.command()
def compare(
    suite: Path = typer.Option(..., exists=True, help="Path to a suite YAML."),
    backend: Backend | None = typer.Option(None, help="Only run this backend."),
    results_dir: Path = typer.Option(DEFAULT_RESULTS, help="Where to write results."),
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
        fail(exc)

    if backend is not None:
        specs = [s for s in specs if s.backend is backend]
    if not specs:
        fail(ConfigError("no runs match the requested backend filter"))

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

    console.print(metrics_table(results))
    console.print(f"\n[green]report:[/] {paths.report_html}")
