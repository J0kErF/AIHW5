"""Ollama backend tests.

The availability path is tested without a daemon (it must skip, not crash). A
real generation test is gated behind ``needs_ollama`` + a reachable daemon.
"""

from __future__ import annotations

import pytest

from localforge.backends.ollama_backend import OllamaBackend
from localforge.backends.runner import run_spec
from localforge.core.capabilities import probe_ollama
from localforge.core.types import Backend, RunSpec

_DAEMON_UP, _ = probe_ollama()


def test_ollama_skips_cleanly_when_unreachable() -> None:
    backend = OllamaBackend()
    backend._base_url = "http://localhost:6553/v1"  # almost certainly closed
    spec = RunSpec(
        model_id="qwen2.5:0.5b",
        backend=Backend.OLLAMA,
        prompt="hi",
        max_new_tokens=8,
    )
    result = run_spec(spec, backend=backend)
    assert result.backend_available is False
    assert "ollama" in (result.note or "").lower()


@pytest.mark.needs_ollama
@pytest.mark.skipif(not _DAEMON_UP, reason="no Ollama daemon reachable")
def test_ollama_real_generation() -> None:
    spec = RunSpec(
        model_id="qwen2.5:0.5b",
        backend=Backend.OLLAMA,
        prompt="Reply with one word: hello.",
        max_new_tokens=8,
    )
    result = run_spec(spec)
    # Either it generated, or it skipped because the tag is not pulled — both are
    # acceptable; what must never happen is a crash.
    assert isinstance(result.backend_available, bool)
    if result.backend_available:
        assert result.text.strip() != ""
