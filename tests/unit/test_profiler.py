"""Tests for the Profiler and baseline."""

from __future__ import annotations

import time

from localforge.profiling.baseline import collect_baseline
from localforge.profiling.probes import current_rss_mb
from localforge.profiling.profiler import Profiler


def test_probes_return_positive_rss() -> None:
    assert current_rss_mb() > 0


def test_profiler_basic_marks() -> None:
    with Profiler(sample_interval_s=0.005) as prof:
        time.sleep(0.02)
        prof.mark_loaded()
        for i in range(4):
            time.sleep(0.01)
            if i == 0:
                prof.mark_prefill_done()
            prof.count_token()

    assert prof.load_s >= 0.015
    assert prof.prefill_ms > 0
    assert prof.decode_tok_s is not None
    assert prof.decode_tok_s > 0
    assert prof.peak_ram_mb > 0
    # CPU-only host: VRAM must be None, never an error.
    assert prof.peak_vram_mb is None or prof.peak_vram_mb >= 0


def test_profiler_decode_none_when_fewer_than_two_tokens() -> None:
    with Profiler() as prof:
        prof.mark_loaded()
        prof.mark_prefill_done()
        prof.count_token()  # only one token
    assert prof.decode_tok_s is None


def test_profiler_does_not_suppress_exceptions() -> None:
    raised = False
    try:
        with Profiler() as prof:
            prof.mark_loaded()
            raise ValueError("boom")
    except ValueError:
        raised = True
    assert raised


def test_baseline_has_expected_keys() -> None:
    record = collect_baseline()
    for key in (
        "host",
        "idle_process_rss_mb",
        "system_available_mb",
        "idle_vram_mb",
    ):
        assert key in record
    assert record["idle_process_rss_mb"] > 0
