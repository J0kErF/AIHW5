"""Test the roofline figure renders."""

from __future__ import annotations

from pathlib import Path

from localforge.reporting.roofline import render_roofline_png


def test_roofline_renders(tmp_path: Path) -> None:
    out = render_roofline_png(tmp_path / "roofline.png")
    assert out.exists()
    assert out.stat().st_size > 0
