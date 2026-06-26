"""PEFT adapter configuration: LoRA / QLoRA / OLoRA (docs/SPECIFICATION.md §3.5).

All three learn a small low-rank update ``W = W0 + BA`` and freeze the base
weights. They differ in setup:

- **LoRA**  — adapters over a full-precision base.
- **QLoRA** — adapters over a 4-bit (NF4) quantized base; needs bitsandbytes+CUDA.
- **OLoRA** — LoRA with orthonormal (QR) adapter init; PEFT-native when available,
  else our fallback (:mod:`localforge.finetune.olora`).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from localforge.core.capabilities import probe_bitsandbytes

DEFAULT_TARGET_MODULES = ["q_proj", "v_proj"]


class FineTuneMethod(StrEnum):
    LORA = "lora"
    QLORA = "qlora"
    OLORA = "olora"


def build_lora_config(
    method: FineTuneMethod,
    *,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> tuple[Any, bool]:
    """Return ``(LoraConfig, used_native_olora)`` for the chosen method.

    ``used_native_olora`` tells the trainer whether it still needs to apply the
    QR-orthonormal fallback after wrapping the model.
    """
    from peft import LoraConfig

    kwargs: dict[str, Any] = {
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules or DEFAULT_TARGET_MODULES,
        "task_type": "CAUSAL_LM",
    }

    used_native_olora = False
    if method is FineTuneMethod.OLORA:
        try:
            config = LoraConfig(init_lora_weights="olora", **kwargs)
            used_native_olora = True
            return config, used_native_olora
        except (ValueError, TypeError):
            # Older PEFT without native OLoRA: build plain LoRA, fallback init later.
            pass

    return LoraConfig(**kwargs), used_native_olora


def quantization_config_for(method: FineTuneMethod) -> tuple[Any, str | None]:
    """Return ``(BitsAndBytesConfig | None, note)`` for QLoRA's 4-bit base.

    On a CPU-only host (no bitsandbytes/CUDA) returns ``(None, note)`` so the
    trainer can fall back to a full-precision base with an explicit note.
    """
    if method is not FineTuneMethod.QLORA:
        return None, None

    ok, reason = probe_bitsandbytes()
    if not ok:
        return None, f"QLoRA 4-bit base unsupported here ({reason}); using fp32 base"

    import torch
    from transformers import BitsAndBytesConfig

    config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    return config, None


def trainable_summary(model: Any) -> dict[str, int]:
    """Count trainable vs total parameters (only A/B should be trainable)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}
