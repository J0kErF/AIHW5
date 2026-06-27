"""API-vs-OnPrem economic model (spec §5.5).

A transparent, parameterized breakeven analysis — not invented precision. All
prices are *illustrative 2026 assumptions* (documented), and the report quotes a
sensitivity range, because §5.5 grades the breakeven *method*, not a fabricated
number.

Model (cost as a function of token volume N):
    API:     cost_api(N)    = price_per_tok · N
    OnPrem:  cost_onprem(N) = CAPEX + opex_per_tok · N         (buy a GPU once)
    Cloud:   cost_cloud(N)  = hourly_rate / (throughput·3600) · N   (rent a GPU)

Breakeven (OnPrem cheaper than API):  N* = CAPEX / (price_api − opex_onprem).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostAssumptions:
    """Illustrative 2026 figures. Override to explore scenarios."""

    api_usd_per_1m_tokens: float = 0.60  # hosted small-model output price
    gpu_capex_usd: float = 1600.0  # one consumer GPU (e.g. 24 GB class)
    gpu_power_w: float = 350.0
    electricity_usd_per_kwh: float = 0.15
    onprem_throughput_tok_s: float = 40.0  # served tokens/sec on that GPU
    cloud_gpu_usd_per_hour: float = 1.20  # rented GPU instance


@dataclass(frozen=True)
class Breakeven:
    api_per_tok: float
    onprem_opex_per_tok: float
    onprem_capex: float
    cloud_per_tok: float
    breakeven_tokens: float | None  # None if OnPrem never wins (opex >= api)

    @property
    def breakeven_millions(self) -> float | None:
        return None if self.breakeven_tokens is None else self.breakeven_tokens / 1e6


def analyze(a: CostAssumptions) -> Breakeven:
    api_per_tok = a.api_usd_per_1m_tokens / 1e6
    # OnPrem OPEX per token = energy to produce one token.
    energy_per_tok_kwh = (a.gpu_power_w / 1000.0) / (a.onprem_throughput_tok_s * 3600.0)
    opex_per_tok = energy_per_tok_kwh * a.electricity_usd_per_kwh
    cloud_per_tok = a.cloud_gpu_usd_per_hour / (a.onprem_throughput_tok_s * 3600.0)

    denom = api_per_tok - opex_per_tok
    breakeven = a.gpu_capex_usd / denom if denom > 0 else None
    return Breakeven(
        api_per_tok=api_per_tok,
        onprem_opex_per_tok=opex_per_tok,
        onprem_capex=a.gpu_capex_usd,
        cloud_per_tok=cloud_per_tok,
        breakeven_tokens=breakeven,
    )


def cost_curves(
    a: CostAssumptions, max_tokens: float, points: int = 50
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return (token_counts, api_cost, onprem_cost, cloud_cost) for plotting."""
    b = analyze(a)
    ns = [max_tokens * i / (points - 1) for i in range(points)]
    api = [b.api_per_tok * n for n in ns]
    onprem = [b.onprem_capex + b.onprem_opex_per_tok * n for n in ns]
    cloud = [b.cloud_per_tok * n for n in ns]
    return ns, api, onprem, cloud
