"""Unit test for the AirLLM backend load/generate (airllm module mocked)."""

from __future__ import annotations

import sys
import types

import pytest
import torch

from localforge.backends.airllm_backend import AirLLMBackend
from localforge.core.types import Backend, RunSpec
from tests._fakes import FakeTokenizer


def _spec() -> RunSpec:
    return RunSpec(model_id="org/m", backend=Backend.AIRLLM, prompt="explain", max_new_tokens=4)


def test_airllm_load_and_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAirModel:
        def __init__(self) -> None:
            self.tokenizer = FakeTokenizer()

        def generate(self, ids: object, **kw: object) -> torch.Tensor:
            return torch.tensor([[1, 2, 3, 4]])

    fake = types.ModuleType("airllm")
    fake.AutoModel = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *a, **k: FakeAirModel()
    )
    monkeypatch.setitem(sys.modules, "airllm", fake)

    backend = AirLLMBackend()
    backend.load(_spec())
    assert backend._model is not None
    assert backend.note is not None and "AirLLM" in backend.note

    out = "".join(backend.generate(_spec()))
    assert out == "decoded text"
