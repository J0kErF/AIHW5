"""Tests for capability probes and settings precedence.

Probes must never raise and must return a (bool, reason) pair regardless of what
is installed on the host running the tests.
"""

from __future__ import annotations

import sys
import types

import pytest

import localforge.core.capabilities as cap
from localforge.config.settings import load_settings
from localforge.core.capabilities import (
    Capabilities,
    probe_airllm,
    probe_bitsandbytes,
    probe_capabilities,
    probe_cuda,
    probe_ollama,
)


def _is_probe(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], bool)
        and isinstance(value[1], str)
        and bool(value[1])
    )


def test_individual_probes_return_probe_pairs() -> None:
    assert _is_probe(probe_cuda())
    assert _is_probe(probe_bitsandbytes())
    assert _is_probe(probe_airllm())
    # Use an almost-certainly-closed port so the probe reports unavailable, not error.
    assert _is_probe(probe_ollama("http://localhost:6553/v1"))


def test_probe_ollama_on_closed_port_is_unavailable() -> None:
    available, reason = probe_ollama("http://localhost:6553/v1")
    assert available is False
    assert "ollama" in reason.lower()


def test_probe_capabilities_aggregates() -> None:
    caps = probe_capabilities("http://localhost:6553/v1")
    assert isinstance(caps, Capabilities)
    d = caps.as_dict()
    assert set(d) == {"cuda", "bitsandbytes", "airllm", "ollama"}
    for entry in d.values():
        assert isinstance(entry["available"], bool)
        assert isinstance(entry["reason"], str)


def test_settings_load_defaults_from_toml() -> None:
    settings = load_settings()
    assert settings.ollama_base_url.endswith("/v1")
    assert settings.airllm_ram_ceiling_mb == 4096
    assert settings.default_model


def _fake_torch(available: bool, *, raises: bool = False) -> types.ModuleType:
    mod = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            if raises:
                raise RuntimeError("broken driver")
            return available

        @staticmethod
        def device_count() -> int:
            return 2

    mod.cuda = _Cuda()  # type: ignore[attr-defined]
    return mod


def test_probe_cuda_reports_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cap, "_has_module", lambda name: True)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True))
    ok, reason = cap.probe_cuda()
    assert ok is True
    assert "2" in reason


def test_probe_cuda_handles_driver_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cap, "_has_module", lambda name: True)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False, raises=True))
    ok, reason = cap.probe_cuda()
    assert ok is False
    assert "failed" in reason.lower()


def test_probe_bitsandbytes_present_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cap, "_has_module", lambda name: name == "bitsandbytes")
    monkeypatch.setattr(cap, "probe_cuda", lambda: (False, "no CUDA device"))
    ok, reason = probe_bitsandbytes()
    assert ok is False
    assert "no CUDA" in reason


def test_settings_override_wins_and_token_is_redacted() -> None:
    settings = load_settings(seed=123, hf_token="super-secret-token")
    assert settings.seed == 123
    # SecretStr keeps the value out of repr/str.
    assert "super-secret-token" not in repr(settings)
    assert settings.hf_token is not None
    assert settings.hf_token.get_secret_value() == "super-secret-token"
