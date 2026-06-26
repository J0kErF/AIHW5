"""Lightweight fakes for transformers, so model-dependent code is unit testable
without downloads or a GPU. Not collected as tests (no test_ prefix)."""

from __future__ import annotations

import sys
import types
from typing import Any

import torch
from torch import nn


class FakeTokenizer:
    chat_template = "fake-template"
    eos_token_id = 0
    pad_token: str | None = None

    def __call__(self, text: Any, return_tensors: Any = None, **kw: Any) -> dict[str, Any]:
        ids = torch.tensor([[1, 2, 3]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def apply_chat_template(
        self,
        messages: Any,
        add_generation_prompt: bool = False,
        return_tensors: Any = None,
        return_dict: bool = False,
        tokenize: bool = True,
    ) -> Any:
        if not tokenize:
            return "formatted text"
        ids = torch.tensor([[1, 2, 3]])
        if return_dict:
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        return ids

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        return "decoded text"

    def save_pretrained(self, path: Any) -> None:
        pass


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _Inner(nn.Module):
    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block() for _ in range(n)])


class FakeCausalLM(nn.Module):
    def __init__(self, n_layers: int = 3) -> None:
        super().__init__()
        self.model = _Inner(n_layers)
        self.lm_head = nn.Linear(4, 4)

    def eval(self) -> FakeCausalLM:  # type: ignore[override]
        return self

    def to(self, device: Any) -> FakeCausalLM:  # type: ignore[override]
        return self

    def generate(self, **kwargs: Any) -> torch.Tensor:
        # Run a dummy forward through the blocks so paging forward-hooks fire.
        x = torch.randn(1, 4)
        for block in self.model.layers:
            x = block(x)
        return torch.tensor([[1, 2, 3, 4, 5]])


def install_fake_transformers(monkeypatch: Any, *, n_layers: int = 3) -> None:
    """Inject a fake ``transformers`` module exposing the symbols our code imports."""
    mod = types.ModuleType("transformers")
    mod.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer())
    mod.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: FakeCausalLM(n_layers)
    )
    mod.BitsAndBytesConfig = lambda **k: {"quant": k}
    monkeypatch.setitem(sys.modules, "transformers", mod)
