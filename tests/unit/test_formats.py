"""Offline tests for format detection and the model registry (no downloads)."""

from __future__ import annotations

from pathlib import Path

from localforge.models.formats import ModelFormat, detect_format, weight_bytes
from localforge.models.registry import ModelInfo, ModelRegistry


def test_detect_safetensors_preferred_over_bin() -> None:
    files = ["config.json", "model.safetensors", "pytorch_model.bin"]
    assert detect_format(files) is ModelFormat.SAFETENSORS


def test_detect_gguf() -> None:
    assert detect_format(["qwen2.5-0.5b-q4_k_m.gguf"]) is ModelFormat.GGUF


def test_detect_pytorch_bin() -> None:
    assert detect_format(["config.json", "pytorch_model.bin"]) is ModelFormat.PYTORCH_BIN


def test_detect_unknown() -> None:
    assert detect_format(["config.json", "tokenizer.json"]) is ModelFormat.UNKNOWN


def test_weight_bytes_sums_only_weights(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "extra.gguf").write_bytes(b"y" * 50)
    (tmp_path / "config.json").write_bytes(b"{}")  # not a weight file
    assert weight_bytes(tmp_path) == 150


def test_registry_round_trip(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path)
    info = ModelInfo("org/model", ModelFormat.SAFETENSORS, 2048, str(tmp_path / "m"))
    reg.record(info)

    got = reg.get("org/model")
    assert got is not None
    assert got.model_id == "org/model"
    assert got.format is ModelFormat.SAFETENSORS
    assert got.size_bytes == 2048
    assert abs(got.size_mb - 2048 / (1024 * 1024)) < 1e-9
    assert [m.model_id for m in reg.all()] == ["org/model"]


def test_registry_missing_returns_none(tmp_path: Path) -> None:
    assert ModelRegistry(tmp_path).get("nope") is None
