"""Capture a real per-block execution trace from a transformers model.

Loads the model, instruments every transformer block with a forward pre-hook,
and runs a short greedy generation. Each block execution emits a FAULT event, so
the resulting stream is the genuine layer-by-layer order across prefill + decode
— an empirical counterpart to the synthesized AirLLM trace (docs/RE_AIRLLM.md).
"""

from __future__ import annotations

from typing import Any

from localforge.core.types import PagingEvent
from localforge.paging.airllm_hook import instrument_layers
from localforge.paging.events import MemorySink
from localforge.paging.tracer import PagingTracer


def capture_transformers_trace(
    model_id: str,
    prompt: str,
    *,
    max_new_tokens: int = 16,
    seed: int = 0,
) -> list[PagingEvent]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer: Any = AutoTokenizer.from_pretrained(model_id)
    model: Any = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()

    tracer = PagingTracer()
    sink = MemorySink()
    tracer.subscribe(sink)
    instrument_layers(model, tracer)
    tracer.start()

    if getattr(tokenizer, "chat_template", None):
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        enc = tokenizer(prompt, return_tensors="pt")

    inputs: dict[str, Any] = dict(enc)
    torch.manual_seed(seed)
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return sink.events
