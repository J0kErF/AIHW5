"""Low-level system probes for the profiler.

Memory readings come from psutil (process RSS) and torch (CUDA). All CUDA
helpers degrade to ``None`` on a CPU-only host (docs/SPECIFICATION.md §3.3).
"""

from __future__ import annotations

import psutil

_PROCESS = psutil.Process()
_MB = 1024 * 1024


def current_rss_mb() -> float:
    """Resident set size of this process, in MB."""
    return float(_PROCESS.memory_info().rss) / _MB


def system_available_mb() -> float:
    """System RAM currently available, in MB."""
    return float(psutil.virtual_memory().available) / _MB


def _cuda() -> object | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch
    except Exception:
        return None
    return None


def reset_cuda_peak() -> None:
    """Reset CUDA peak-memory tracking, if a GPU is present."""
    torch = _cuda()
    if torch is not None:
        torch.cuda.reset_peak_memory_stats()  # type: ignore[attr-defined]


def cuda_peak_mb() -> float | None:
    """Peak CUDA memory allocated since the last reset, or ``None`` on CPU."""
    torch = _cuda()
    if torch is None:
        return None
    return float(torch.cuda.max_memory_allocated()) / _MB  # type: ignore[attr-defined]


def cuda_current_mb() -> float | None:
    """Currently allocated CUDA memory, or ``None`` on CPU."""
    torch = _cuda()
    if torch is None:
        return None
    return float(torch.cuda.memory_allocated()) / _MB  # type: ignore[attr-defined]
