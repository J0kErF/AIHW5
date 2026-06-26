"""End-to-end report generation test (no models, no GPU)."""

from __future__ import annotations

from pathlib import Path

from localforge.core.types import Backend, RunResult
from localforge.reporting.report import write_report
from localforge.reporting.store import ResultStore


def _ok(backend: Backend, h: str, ram: float, tok: float) -> RunResult:
    return RunResult(
        spec_hash=h,
        backend=backend,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        text="virtual memory ...",
        load_s=2.0,
        prefill_ms=120.0,
        decode_tok_s=tok,
        peak_ram_mb=ram,
        backend_available=True,
    )


def _skipped(backend: Backend, h: str) -> RunResult:
    return RunResult(
        spec_hash=h,
        backend=backend,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend_available=False,
        note="no daemon at localhost:11434",
    )


def test_write_report_produces_all_artifacts(tmp_path: Path) -> None:
    results = [
        _ok(Backend.TRANSFORMERS, "h1", 900.0, 12.0),
        _skipped(Backend.OLLAMA, "h2"),
        _ok(Backend.AIRLLM, "h3", 350.0, 3.0),
    ]
    store = ResultStore(tmp_path / "results")
    store.save_suite("demo", results)
    run = store.load_suite("demo")
    assert run is not None

    paths = write_report("demo", results, run, tmp_path / "results" / "reports")

    assert paths.report_html.exists()
    assert paths.matrix_md.exists()
    assert paths.matrix_json.exists()
    assert len(paths.charts) >= 1  # at least peak RAM chart for the 2 ok runs

    html = paths.report_html.read_text(encoding="utf-8")
    assert "localforge" in html
    assert "skipped" in html  # the Ollama row is shown, not dropped
    assert "no daemon" in html

    md = paths.matrix_md.read_text(encoding="utf-8")
    assert "Backend" in md and "transformers" in md and "airllm" in md


def test_report_handles_all_skipped(tmp_path: Path) -> None:
    results = [_skipped(Backend.OLLAMA, "h1"), _skipped(Backend.AIRLLM, "h2")]
    store = ResultStore(tmp_path / "results")
    store.save_suite("empty", results)
    run = store.load_suite("empty")
    assert run is not None

    paths = write_report("empty", results, run, tmp_path / "reports")
    assert paths.report_html.exists()
    assert paths.charts == []  # nothing successful to chart
