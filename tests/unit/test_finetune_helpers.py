"""Unit tests for finetune generate/format helpers, OLoRA, and adapter configs."""

from __future__ import annotations

import torch
from torch import nn

from localforge.finetune.dataset import Example
from localforge.finetune.generate import format_example, generate_reply
from localforge.finetune.olora import apply_olora_init
from tests._fakes import FakeCausalLM, FakeTokenizer


def test_format_example_uses_chat_template() -> None:
    out = format_example(FakeTokenizer(), Example("q", "a"))
    assert out == "formatted text"


def test_format_example_without_chat_template() -> None:
    tok = FakeTokenizer()
    tok.chat_template = None  # type: ignore[assignment]
    out = format_example(tok, Example("question", "answer"))
    assert out == "question\nanswer"


def test_generate_reply_returns_decoded_text() -> None:
    assert generate_reply(FakeCausalLM(), FakeTokenizer(), "prompt") == "decoded text"


def test_apply_olora_init_orthonormalizes_lora_a() -> None:
    # A module exposing a lora_A weight, as PEFT would.
    class WithAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A = nn.Linear(4, 8, bias=False)  # name contains 'lora_A'

    model = WithAdapter()
    count = apply_olora_init(model)
    assert count == 1
    w = dict(model.named_parameters())["lora_A.weight"]
    gram = w.t() @ w if w.shape[0] >= w.shape[1] else w @ w.t()
    eye = torch.eye(min(w.shape))
    assert torch.allclose(gram, eye, atol=1e-5)
