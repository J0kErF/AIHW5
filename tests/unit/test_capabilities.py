"""Tests for capability probes and settings precedence.

Probes must never raise and must return a (bool, reason) pair regardless of what
is installed on the host running the tests.
"""

from __future__ import annotations

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


def test_settings_override_wins_and_token_is_redacted() -> None:
    settings = load_settings(seed=123, hf_token="super-secret-token")
    assert settings.seed == 123
    # SecretStr keeps the value out of repr/str.
    assert "super-secret-token" not in repr(settings)
    assert settings.hf_token is not None
    assert settings.hf_token.get_secret_value() == "super-secret-token"
