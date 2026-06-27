"""Conceptual roofline figure (spec §3, §5.6).

The roofline frames the two phases of LLM inference:
  - **Prefill** processes the whole prompt as matrix-matrix products: high
    arithmetic intensity, so it lives under the *compute* roof (compute-bound) →
    this is TTFT.
  - **Decode** emits one token at a time as matrix-vector products: it must read
    *every* weight to produce *one* token, so its intensity is ~1 FLOP/byte and
    it lives under the *memory* roof (memory-bound) → this is TPOT.
  - **AirLLM decode** re-streams the weights from disk each token, dropping the
    effective bandwidth from RAM (~tens of GB/s) to NVMe (~GB/s): it sits far
    left, *disk*-bandwidth-bound — which is why its TPOT is orders worse.

Hardware numbers are illustrative (a 4-core laptop), labeled as such.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Illustrative laptop-class figures.
PEAK_GFLOPS = 100.0  # compute roof
RAM_BW_GBs = 50.0  # RAM bandwidth roof
DISK_BW_GBs = 3.0  # NVMe bandwidth (AirLLM's effective roof)


def render_roofline_png(out: Path) -> Path:
    intensities = [10**e for e in [-1.0 + 0.1 * i for i in range(41)]]  # 0.1 .. 1000

    def attainable(bw_gbs: float) -> list[float]:
        return [min(PEAK_GFLOPS, bw_gbs * x) for x in intensities]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(intensities, attainable(RAM_BW_GBs), color="#01133f", lw=2, label="RAM roof (~50 GB/s)")
    ax.plot(
        intensities,
        attainable(DISK_BW_GBs),
        color="#2e8b57",
        lw=2,
        ls="--",
        label="NVMe roof (~3 GB/s, AirLLM)",
    )

    # Phase markers (qualitative positions).
    ax.scatter([80], [PEAK_GFLOPS], color="#cc5500", zorder=5)
    ax.annotate("Prefill (TTFT)\ncompute-bound", (80, PEAK_GFLOPS), (30, PEAK_GFLOPS * 0.5), fontsize=9)
    ax.scatter([1.2], [RAM_BW_GBs * 1.2], color="#cc5500", zorder=5)
    ax.annotate("Decode (TPOT)\nmemory-bound", (1.2, RAM_BW_GBs * 1.2), (1.4, 8), fontsize=9)
    ax.scatter([1.2], [DISK_BW_GBs * 1.2], color="#2e8b57", zorder=5)
    ax.annotate("AirLLM decode\ndisk-bound", (1.2, DISK_BW_GBs * 1.2), (2.0, 1.2), fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("Attainable performance (GFLOP/s)")
    ax.set_title("Roofline: prefill is compute-bound, decode is memory/disk-bound (illustrative)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
