"""Instrumentation sources for the paging tracer (docs/SPECIFICATION.md §3.6).

Two sources feed the visualizer:

1. :func:`instrument_layers` — **real** per-layer execution capture via PyTorch
   forward pre-hooks on each transformer block. AirLLM loads a block's weights
   from disk immediately before running it, so a per-block hook records the true
   streaming order, block sizes, and timing. The same hook works on the
   transformers backend (where blocks stay resident), giving an honest baseline
   to contrast against.

2. :func:`synthesize_airllm_trace` — an explicit **model** of AirLLM's bounded
   working set: with capacity for only a few resident blocks, every forward pass
   faults blocks in from disk and evicts the oldest. Used to demonstrate the
   OS-paging dynamics on machines where ``airllm`` is not installed, and as the
   prediction the real trace is validated against (docs/RE_AIRLLM.md, T15a).
"""

from __future__ import annotations

from collections import deque
from typing import Any

from localforge.core.types import PageAction, PageSource, PagingEvent
from localforge.paging.tracer import PagingTracer


def _find_decoder_layers(model: Any) -> Any:
    """Locate the list of transformer blocks across common architectures."""
    for path in ("model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers"):
        obj: Any = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if obj is not None and len(obj) > 0:
                return obj
        except AttributeError:
            continue
    return None


def _layer_bytes(module: Any) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def instrument_layers(model: Any, tracer: PagingTracer) -> int:
    """Register forward pre-hooks emitting one FAULT event per block execution.

    Returns the number of blocks instrumented (0 if the architecture is
    unrecognized). The caller is responsible for calling ``tracer.start()``.
    """
    layers = _find_decoder_layers(model)
    if layers is None:
        return 0

    for index, block in enumerate(layers):
        n_bytes = _layer_bytes(block)

        def _hook(_module: Any, _args: Any, _idx: int = index, _b: int = n_bytes) -> None:
            tracer.emit(_idx, PageAction.FAULT, _b, PageSource.MMAP)

        block.register_forward_pre_hook(_hook)
    return len(layers)


def synthesize_airllm_trace(
    *,
    n_layers: int = 32,
    layer_mb: int = 400,
    ram_ceiling_mb: int = 4096,
    passes: int = 2,
    fault_ms: float = 9.0,
    hit_ms: float = 0.3,
) -> list[PagingEvent]:
    """Model AirLLM layer streaming as an OS pager with a bounded working set.

    ``passes`` = 1 prefill + (tokens-1) decode passes over all layers. With
    ``capacity < n_layers`` every pass re-faults layers from disk and evicts the
    oldest — the signature behavior the visualizer surfaces.
    """
    capacity = max(1, ram_ceiling_mb // layer_mb)
    layer_bytes = layer_mb * 1024 * 1024
    resident: deque[int] = deque()
    events: list[PagingEvent] = []
    t = 0.0

    for _ in range(passes):
        for layer in range(n_layers):
            if layer in resident:
                events.append(
                    PagingEvent(
                        layer=layer,
                        action=PageAction.HIT,
                        bytes=layer_bytes,
                        source=PageSource.MMAP,
                        t_ms=t,
                    )
                )
                t += hit_ms
                continue
            if len(resident) >= capacity:
                evicted = resident.popleft()
                events.append(
                    PagingEvent(
                        layer=evicted,
                        action=PageAction.EVICT,
                        bytes=layer_bytes,
                        source=PageSource.DISK,
                        t_ms=t,
                    )
                )
            events.append(
                PagingEvent(
                    layer=layer,
                    action=PageAction.FAULT,
                    bytes=layer_bytes,
                    source=PageSource.MMAP,
                    t_ms=t,
                )
            )
            t += fault_ms
            resident.append(layer)

    return events
