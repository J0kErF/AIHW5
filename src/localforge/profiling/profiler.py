"""The Profiler: a context manager that measures one inference run uniformly.

The runner drives it with explicit marks so measurement is identical across all
backends (docs/IMPLEMENTATION.md §2.2):

    with Profiler() as prof:
        backend.load(spec)
        prof.mark_loaded()
        for i, token in enumerate(backend.generate(spec)):
            if i == 0:
                prof.mark_prefill_done()   # first token == end of prefill
            prof.count_token()

Peak RSS is captured by a lightweight background sampler thread so we catch the
transient high-water mark during load and generation, not just the endpoints.
"""

from __future__ import annotations

import threading
from time import perf_counter
from types import TracebackType
from typing import Literal

from localforge.profiling.probes import (
    cuda_peak_mb,
    current_rss_mb,
    reset_cuda_peak,
)


class Profiler:
    def __init__(self, sample_interval_s: float = 0.02) -> None:
        self._interval = sample_interval_s
        # Results (populated on exit / via marks).
        self.load_s: float = 0.0
        self.prefill_ms: float = 0.0
        self.decode_tok_s: float | None = None
        self.peak_ram_mb: float = 0.0
        self.peak_vram_mb: float | None = None

    def __enter__(self) -> Profiler:
        reset_cuda_peak()
        self._t0 = perf_counter()
        self._t_load: float | None = None
        self._t_first: float | None = None
        self._t_last: float | None = None
        self._tokens = 0
        self._peak_rss = current_rss_mb()
        self._stop = threading.Event()
        self._sampler = threading.Thread(target=self._sample, daemon=True)
        self._sampler.start()
        return self

    def _sample(self) -> None:
        while not self._stop.wait(self._interval):
            rss = current_rss_mb()
            if rss > self._peak_rss:
                self._peak_rss = rss

    def mark_loaded(self) -> None:
        """Call once the model weights are loaded; fixes ``load_s``."""
        self._t_load = perf_counter()
        self.load_s = self._t_load - self._t0

    def mark_prefill_done(self) -> None:
        """Call when the first token is produced; fixes ``prefill_ms``."""
        now = perf_counter()
        base = self._t_load if self._t_load is not None else self._t0
        self.prefill_ms = (now - base) * 1000.0
        self._t_first = now
        self._t_last = now

    def count_token(self) -> None:
        """Call once per generated token."""
        self._tokens += 1
        self._t_last = perf_counter()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        self._stop.set()
        self._sampler.join(timeout=1.0)
        self._peak_rss = max(self._peak_rss, current_rss_mb())
        self.peak_ram_mb = self._peak_rss
        self.peak_vram_mb = cuda_peak_mb()
        # Decode throughput = tokens generated after the first, over the decode window.
        if self._t_first is not None and self._t_last is not None and self._tokens >= 2:
            window = self._t_last - self._t_first
            self.decode_tok_s = (self._tokens - 1) / window if window > 0 else None
        else:
            self.decode_tok_s = None
        return False  # never suppress exceptions
