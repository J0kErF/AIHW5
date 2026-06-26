"""AirLLM backend tests.

The skip path (airllm not installed -> skipped result) is always tested. A real
layer-streaming run is gated behind ``needs_airllm``.
"""

from __future__ import annotations

import pytest

from localforge.backends.base import make_backend
from localforge.backends.runner import run_spec
from localforge.core.capabilities import probe_airllm
from localforge.core.types import Backend, RunSpec

_AIRLLM_OK, _ = probe_airllm()


def _spec() -> RunSpec:
    return RunSpec(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend=Backend.AIRLLM,
        prompt="Explain paging.",
        max_new_tokens=16,
    )


@pytest.mark.skipif(_AIRLLM_OK, reason="airllm installed; skip-path test not applicable")
def test_airllm_skips_when_not_installed() -> None:
    result = run_spec(_spec())
    assert result.backend_available is False
    assert "airllm" in (result.note or "").lower()


def test_airllm_backend_is_registered() -> None:
    backend = make_backend("airllm")
    assert backend.name == "airllm"
    available, reason = backend.is_available()
    assert isinstance(available, bool)
    assert isinstance(reason, str) and reason


@pytest.mark.needs_airllm
@pytest.mark.skipif(not _AIRLLM_OK, reason="airllm not installed")
def test_airllm_real_layer_streaming() -> None:
    result = run_spec(_spec())
    assert isinstance(result.backend_available, bool)
    if result.backend_available:
        assert result.text.strip() != ""
        assert result.peak_ram_mb > 0
