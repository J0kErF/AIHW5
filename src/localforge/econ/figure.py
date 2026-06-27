"""Breakeven figure for the economic analysis (spec §5.5)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from localforge.econ.breakeven import CostAssumptions, analyze, cost_curves  # noqa: E402


def render_breakeven_png(
    assumptions: CostAssumptions,
    out: Path,
    max_tokens: float = 5e9,
) -> Path:
    """Plot cumulative cost vs token volume for API / OnPrem / Cloud."""
    b = analyze(assumptions)
    ns, api, onprem, cloud = cost_curves(assumptions, max_tokens)

    fig, ax = plt.subplots(figsize=(8, 5))
    billions = [n / 1e9 for n in ns]
    ax.plot(billions, api, label="API (pay per token)", color="#cc5500", lw=2)
    ax.plot(billions, onprem, label="OnPrem (buy GPU)", color="#01133f", lw=2)
    ax.plot(billions, cloud, label="Cloud GPU (rent)", color="#2e8b57", lw=2, ls="--")

    if b.breakeven_tokens is not None and b.breakeven_tokens <= max_tokens:
        x = b.breakeven_tokens / 1e9
        y = b.onprem_capex + b.onprem_opex_per_tok * b.breakeven_tokens
        ax.axvline(x, color="#888", ls=":", lw=1)
        ax.annotate(
            f"breakeven ≈ {b.breakeven_millions:.0f}M tokens",
            xy=(x, y),
            xytext=(x * 1.05, y * 1.25),
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "#888"},
        )

    ax.set_xlabel("Tokens served (billions)")
    ax.set_ylabel("Cumulative cost (USD)")
    ax.set_title("API vs OnPrem vs Cloud GPU — cost over volume (illustrative)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
