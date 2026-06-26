"""Hugging Face ``transformers`` backend (docs/SPECIFICATION.md §3.2).

Full-residency inference: the whole model is loaded into RAM/VRAM, then greedy
generation streams token-by-token via ``TextIteratorStreamer`` so the runner can
separate prefill (time to first token) from decode (subsequent tokens).

CPU-first: CUDA-only paths (fp16, NF4 4-bit) fall back to fp32 with an explicit
note rather than failing (docs/DECISIONS.md D1).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

from localforge.backends.base import register
from localforge.core.capabilities import probe_bitsandbytes, probe_cuda
from localforge.core.types import Device, Dtype, RunSpec


@register
class TransformersBackend:
    name = "transformers"

    def __init__(self) -> None:
        self.note: str | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"

    def is_available(self) -> tuple[bool, str]:
        # torch + transformers are core dependencies, so this backend always runs.
        return (True, "transformers (core dependency)")

    def _resolve(self, spec: RunSpec) -> tuple[str, Any, dict[str, Any]]:
        """Resolve (device, torch_dtype, extra from_pretrained kwargs)."""
        import torch

        notes: list[str] = []
        cuda_ok, cuda_reason = probe_cuda()
        want_cuda = spec.device in (Device.AUTO, Device.CUDA)
        device = "cuda" if (cuda_ok and want_cuda) else "cpu"
        if spec.device is Device.CUDA and not cuda_ok:
            notes.append(f"CUDA requested but {cuda_reason}; using CPU")

        extra: dict[str, Any] = {}
        dtype: Any = torch.float32

        if spec.dtype is Dtype.NF4:
            bb_ok, bb_reason = probe_bitsandbytes()
            if bb_ok:
                from transformers import BitsAndBytesConfig

                extra["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                extra["device_map"] = "auto"
                return device, None, extra  # dtype managed by the quant config
            notes.append(f"NF4 unsupported here ({bb_reason}); using fp32")
        elif spec.dtype is Dtype.FP16:
            if device == "cuda":
                dtype = torch.float16
            else:
                notes.append("fp16 is unreliable on CPU; using fp32")
        elif spec.dtype is Dtype.BF16:
            dtype = torch.bfloat16

        self.note = "; ".join(notes) if notes else None
        return device, dtype, extra

    def load(self, spec: RunSpec) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device, dtype, extra = self._resolve(spec)
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        model_kwargs: dict[str, Any] = dict(extra)
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        self._model = AutoModelForCausalLM.from_pretrained(spec.model_id, **model_kwargs)
        if "device_map" not in model_kwargs:  # not auto-placed by a quant config
            self._model.to(device)
        self._model.eval()

    def _encode(self, spec: RunSpec) -> dict[str, Any]:
        # transformers 5.x returns a BatchEncoding (dict) here, not a raw tensor,
        # so we keep input_ids *and* attention_mask and move both to the device.
        tok = self._tokenizer
        if getattr(tok, "chat_template", None):
            messages = [{"role": "user", "content": spec.prompt}]
            enc = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            enc = tok(spec.prompt, return_tensors="pt")
        return {k: v.to(self._device) for k, v in enc.items()}

    def generate(self, spec: RunSpec) -> Iterator[str]:
        import torch
        from transformers import TextIteratorStreamer

        torch.manual_seed(spec.seed)
        inputs = self._encode(spec)
        # timeout guards against a stalled generation thread deadlocking the iterator.
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300
        )
        gen_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": spec.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self._tokenizer.eos_token_id,
            "streamer": streamer,
        }
        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()
        try:
            for piece in streamer:
                if piece:
                    yield piece
        finally:
            thread.join()
