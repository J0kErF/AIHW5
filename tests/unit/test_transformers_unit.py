"""Unit tests for the transformers backend resolve/load logic (no real model)."""

from __future__ import annotations

import pytest
import torch

from localforge.backends.transformers_backend import TransformersBackend
from localforge.core.types import Backend, Device, Dtype, RunSpec
from tests._fakes import install_fake_transformers


def _spec(**kw: object) -> RunSpec:
    base: dict[str, object] = {
        "model_id": "m",
        "backend": Backend.TRANSFORMERS,
        "prompt": "hi",
        "max_new_tokens": 8,
    }
    base.update(kw)
    return RunSpec(**base)  # type: ignore[arg-type]


def test_is_available() -> None:
    ok, reason = TransformersBackend().is_available()
    assert ok is True
    assert "transformers" in reason


def test_resolve_fp32_cpu() -> None:
    device, dtype, extra = TransformersBackend()._resolve(_spec(dtype=Dtype.FP32))
    assert device == "cpu"
    assert dtype is torch.float32
    assert extra == {}


def test_resolve_fp16_on_cpu_falls_back() -> None:
    bk = TransformersBackend()
    _, dtype, _ = bk._resolve(_spec(dtype=Dtype.FP16))
    assert dtype is torch.float32
    assert bk.note is not None and "fp16" in bk.note


def test_resolve_bf16() -> None:
    _, dtype, _ = TransformersBackend()._resolve(_spec(dtype=Dtype.BF16))
    assert dtype is torch.bfloat16


def test_resolve_nf4_on_cpu_falls_back_with_note() -> None:
    bk = TransformersBackend()
    _, _, _ = bk._resolve(_spec(dtype=Dtype.NF4))
    assert bk.note is not None and "NF4" in bk.note


def test_resolve_cuda_requested_without_gpu_notes() -> None:
    bk = TransformersBackend()
    device, _, _ = bk._resolve(_spec(device=Device.CUDA))
    assert device == "cpu"
    assert bk.note is not None and "CUDA" in bk.note


def test_load_with_fake_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_transformers(monkeypatch)
    bk = TransformersBackend()
    bk.load(_spec())
    assert bk._model is not None
    assert bk._device == "cpu"
