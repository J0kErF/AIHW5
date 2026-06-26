"""Run a single RunSpec under the Profiler and produce a RunResult.

This is the one place measurement happens, so every backend is profiled
identically. Unavailable backends become non-fatal skipped results; only the
backend boundary is caught here (docs/IMPLEMENTATION.md §7.2).
"""

from __future__ import annotations

from localforge.backends.base import InferenceBackend, make_backend
from localforge.core.errors import BackendUnavailable
from localforge.core.hashing import spec_hash
from localforge.core.types import RunResult, RunSpec
from localforge.profiling.profiler import Profiler


def run_spec(spec: RunSpec, backend: InferenceBackend | None = None) -> RunResult:
    """Execute ``spec`` and return its measured result.

    ``backend`` may be supplied directly (used by tests); otherwise it is
    resolved from ``spec.backend`` via the registry factory.
    """
    bk = backend if backend is not None else make_backend(spec.backend.value)
    h = spec_hash(spec)

    ok, reason = bk.is_available()
    if not ok:
        return RunResult.skipped(spec, h, reason)

    try:
        with Profiler() as prof:
            bk.load(spec)
            prof.mark_loaded()
            pieces: list[str] = []
            for i, token in enumerate(bk.generate(spec)):
                if i == 0:
                    prof.mark_prefill_done()
                prof.count_token()
                pieces.append(token)
            text = "".join(pieces)
    except BackendUnavailable as exc:
        return RunResult.skipped(spec, h, exc.reason)

    return RunResult(
        spec_hash=h,
        backend=spec.backend,
        model_id=spec.model_id,
        text=text,
        load_s=prof.load_s,
        prefill_ms=prof.prefill_ms,
        decode_tok_s=prof.decode_tok_s,
        peak_ram_mb=prof.peak_ram_mb,
        peak_vram_mb=prof.peak_vram_mb,
        backend_available=True,
    )
