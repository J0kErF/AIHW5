"""Assemble the comparison report (docs/SPECIFICATION.md §3.4.1).

Writes, under ``results/reports/<suite_id>/``: matrix.md, matrix.json, the PNG
charts, and a self-contained report.html bundling the table, charts, and
reproducibility metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from localforge.core.types import RunResult
from localforge.reporting.charts import render_charts
from localforge.reporting.matrix import build_rows, to_json, to_markdown
from localforge.reporting.store import SuiteRun

_TEMPLATES = Path(__file__).resolve().parents[1] / "templates"


@dataclass
class ReportPaths:
    report_html: Path
    matrix_md: Path
    matrix_json: Path
    charts: list[Path]


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=select_autoescape(["html"]),
    )


def write_report(
    suite_id: str,
    results: list[RunResult],
    suite_run: SuiteRun,
    reports_dir: Path,
) -> ReportPaths:
    out_dir = reports_dir / suite_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(results)
    matrix_md = out_dir / "matrix.md"
    matrix_md.write_text(to_markdown(rows), encoding="utf-8")
    matrix_json = out_dir / "matrix.json"
    matrix_json.write_text(to_json(rows), encoding="utf-8")

    charts = render_charts(results, out_dir)

    template = _env().get_template("report.html.j2")
    html = template.render(
        suite_id=suite_id,
        rows=rows,
        charts=[p.name for p in charts],
        meta=suite_run,
        n_total=len(results),
        n_ok=sum(1 for r in results if r.backend_available),
    )
    report_html = out_dir / "report.html"
    report_html.write_text(html, encoding="utf-8")

    return ReportPaths(report_html, matrix_md, matrix_json, charts)
