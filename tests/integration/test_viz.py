"""Tests for the paging visualizer (state, static render, export, CLI)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from localforge.cli.app import app
from localforge.core.types import PageAction, PageSource, PagingEvent
from localforge.paging.airllm_hook import synthesize_airllm_trace
from localforge.viz.render import export_paging_artifacts
from localforge.viz.state import PagingViewState
from localforge.viz.tui import plain_summary

runner = CliRunner()
_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "sample_paging.jsonl"


def test_view_state_counts_and_residency() -> None:
    state = PagingViewState()
    state.ingest(
        PagingEvent(layer=0, action=PageAction.FAULT, bytes=100, source=PageSource.MMAP, t_ms=0)
    )
    state.ingest(
        PagingEvent(layer=1, action=PageAction.FAULT, bytes=100, source=PageSource.MMAP, t_ms=1)
    )
    state.ingest(
        PagingEvent(layer=0, action=PageAction.HIT, bytes=100, source=PageSource.MMAP, t_ms=2)
    )
    state.ingest(
        PagingEvent(layer=0, action=PageAction.EVICT, bytes=100, source=PageSource.DISK, t_ms=3)
    )
    assert state.faults == 2
    assert state.hits == 1
    assert state.evicts == 1
    assert state.resident == {1}
    assert state.resident_count == 1
    bar = state.residency_bar(2, width=2)
    assert len(bar) == 2


def test_export_artifacts_writes_three_files(tmp_path: Path) -> None:
    events = synthesize_airllm_trace(n_layers=12, layer_mb=200, ram_ceiling_mb=800, passes=2)
    artifacts = export_paging_artifacts(events, tmp_path / "paging")
    assert artifacts.jsonl.exists()
    assert artifacts.png.exists()
    assert artifacts.html.exists()
    html = artifacts.html.read_text(encoding="utf-8")
    assert "page faults" in html
    assert "mmap" in html.lower()


def test_plain_summary_mentions_key_numbers() -> None:
    events = synthesize_airllm_trace(n_layers=10, layer_mb=200, ram_ceiling_mb=600, passes=2)
    text = plain_summary(events, 10)
    assert "faults" in text
    assert "peak resident" in text
    assert "hierarchy" in text


def test_cli_visualize_replay_no_tui(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "visualize",
            "--replay",
            str(_FIXTURE),
            "--no-tui",
            "--results-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "paging" / "paging.png").exists()
    assert (tmp_path / "paging" / "paging.html").exists()
    assert "faults" in result.output
