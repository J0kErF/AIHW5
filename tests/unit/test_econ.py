"""Tests for the economic breakeven model (spec §5.5)."""

from __future__ import annotations

from pathlib import Path

from localforge.econ.breakeven import CostAssumptions, analyze, cost_curves
from localforge.econ.figure import render_breakeven_png


def test_analyze_has_positive_breakeven() -> None:
    b = analyze(CostAssumptions())
    assert b.api_per_tok > 0
    assert b.onprem_opex_per_tok > 0
    assert b.cloud_per_tok > 0
    assert b.breakeven_tokens is not None and b.breakeven_tokens > 0
    assert b.breakeven_millions is not None


def test_no_breakeven_when_api_is_cheaper_than_opex() -> None:
    # Absurdly cheap API -> OnPrem CAPEX never pays back.
    b = analyze(CostAssumptions(api_usd_per_1m_tokens=1e-6))
    assert b.breakeven_tokens is None
    assert b.breakeven_millions is None


def test_cost_curves_shapes_and_monotonic() -> None:
    a = CostAssumptions()
    ns, api, onprem, cloud = cost_curves(a, 1e9, points=20)
    assert len(ns) == len(api) == len(onprem) == len(cloud) == 20
    assert api[-1] > api[0]
    assert onprem[-1] > onprem[0]
    assert abs(onprem[0] - a.gpu_capex_usd) < 1e-6  # OnPrem starts at CAPEX


def test_figure_renders(tmp_path: Path) -> None:
    out = render_breakeven_png(CostAssumptions(), tmp_path / "breakeven.png")
    assert out.exists()
    assert out.stat().st_size > 0
