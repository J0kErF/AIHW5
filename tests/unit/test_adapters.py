"""Tests for PEFT adapter configuration (offline; no model)."""

from __future__ import annotations

from localforge.finetune.adapters import (
    FineTuneMethod,
    build_lora_config,
    quantization_config_for,
)


def test_lora_config_defaults() -> None:
    config, native_olora = build_lora_config(FineTuneMethod.LORA)
    assert config.r == 8
    assert config.lora_alpha == 16
    assert "q_proj" in config.target_modules
    assert native_olora is False


def test_olora_config_builds_either_way() -> None:
    # Native OLoRA support varies by PEFT version; both branches yield a valid config.
    config, _native = build_lora_config(FineTuneMethod.OLORA)
    assert config.r == 8


def test_qlora_quant_config_falls_back_on_cpu() -> None:
    quant, note = quantization_config_for(FineTuneMethod.QLORA)
    # On a CPU-only host bitsandbytes is unavailable -> (None, explanatory note).
    if quant is None:
        assert note is not None and "fp32" in note


def test_non_qlora_has_no_quant_config() -> None:
    quant, note = quantization_config_for(FineTuneMethod.LORA)
    assert quant is None
    assert note is None
