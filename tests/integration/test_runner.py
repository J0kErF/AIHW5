"""Integration tests for the backend registry + runner."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from localforge.backends.base import (
    UnavailableBackend,
    available_backends,
    make_backend,
)
from localforge.backends.runner import run_spec
from localforge.core.errors import BackendUnavailable, UnknownBackend
from localforge.core.types import Backend, RunSpec


def test_make_fake_backend_runs_end_to_end(fake_spec: RunSpec) -> None:
    result = run_spec(fake_spec)
    assert result.backend_available is True
    assert result.backend is Backend.FAKE
    assert result.text.strip() != ""
    assert result.load_s >= 0
    assert result.prefill_ms >= 0
    assert result.peak_ram_mb > 0
    assert result.peak_vram_mb is None or result.peak_vram_mb >= 0


def test_fake_run_is_deterministic(fake_spec: RunSpec) -> None:
    assert run_spec(fake_spec).text == run_spec(fake_spec).text


def test_unknown_backend_raises() -> None:
    with pytest.raises(UnknownBackend):
        make_backend("does-not-exist")


def test_unavailable_backend_is_skipped_not_crashed(fake_spec: RunSpec) -> None:
    bk = UnavailableBackend("ollama", "no daemon at localhost:11434")
    result = run_spec(fake_spec, backend=bk)
    assert result.backend_available is False
    assert "daemon" in (result.note or "")


def test_backend_unavailable_during_load_is_skipped(fake_spec: RunSpec) -> None:
    class FlakyBackend:
        name = "flaky"

        def is_available(self) -> tuple[bool, str]:
            return (True, "claims available")

        def load(self, spec: RunSpec) -> None:
            raise BackendUnavailable("flaky", "blew up on load")

        def generate(self, spec: RunSpec) -> Iterator[str]:
            yield "unreachable"

    result = run_spec(fake_spec, backend=FlakyBackend())
    assert result.backend_available is False
    assert result.note == "blew up on load"


def test_fake_is_registered() -> None:
    assert "fake" in available_backends()
    assert isinstance(make_backend("fake").is_available(), tuple)
