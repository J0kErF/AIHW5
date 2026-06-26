"""Minimal PEFT fine-tuning loop (docs/SPECIFICATION.md §3.5).

A deliberately small, dependency-light trainer: wrap the base model with a
LoRA/QLoRA/OLoRA adapter, run a handful of gradient steps on a tiny SFT set on
CPU, log a held-out generation before and after to show the adapter changed
behavior, and save the adapter. The point is a correct, reproducible
demonstration of PEFT — not a production trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localforge.config.settings import Settings
from localforge.core.logging import get_logger
from localforge.finetune.adapters import (
    FineTuneMethod,
    build_lora_config,
    quantization_config_for,
    trainable_summary,
)
from localforge.finetune.dataset import Example, load_sft
from localforge.finetune.olora import apply_olora_init

_log = get_logger(__name__)
_HELDOUT = "Define paging in one sentence."


@dataclass
class FineTuneResult:
    method: FineTuneMethod
    model_id: str
    trainable_params: int
    total_params: int
    steps: int
    final_loss: float
    before: str
    after: str
    adapter_path: str
    note: str | None = None

    @property
    def trainable_pct(self) -> float:
        return 100.0 * self.trainable_params / max(self.total_params, 1)


def _format(tokenizer: Any, ex: Example) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "user", "content": ex.instruction},
            {"role": "assistant", "content": ex.response},
        ]
        return str(tokenizer.apply_chat_template(messages, tokenize=False))
    return f"{ex.instruction}\n{ex.response}"


def _generate(model: Any, tokenizer: Any, prompt: str, max_new: int = 16) -> str:
    import torch

    if getattr(tokenizer, "chat_template", None):
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        enc = tokenizer(prompt, return_tensors="pt")
    prompt_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    text: str = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    return text.strip()


def train_adapter(
    model_id: str,
    method: FineTuneMethod,
    data_path: Path,
    settings: Settings,
    *,
    steps: int = 12,
    lr: float = 1e-3,
    max_len: int = 128,
    out_dir: Path | None = None,
) -> FineTuneResult:
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    examples = load_sft(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant, note = quantization_config_for(method)
    model_kwargs: dict[str, Any] = {}
    if quant is not None:
        model_kwargs["quantization_config"] = quant
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["dtype"] = torch.float32
    base = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if quant is not None:
        from peft import prepare_model_for_kbit_training

        base = prepare_model_for_kbit_training(base)  # type: ignore[no-untyped-call]

    config, native_olora = build_lora_config(method)
    model = get_peft_model(base, config)
    if method is FineTuneMethod.OLORA and not native_olora:
        n = apply_olora_init(model)
        note = (f"{note}; " if note else "") + f"OLoRA QR-init applied to {n} adapter matrices"

    summary = trainable_summary(model)
    _log.info(
        "trainable %d / %d params (%.3f%%)",
        summary["trainable"],
        summary["total"],
        100.0 * summary["trainable"] / max(summary["total"], 1),
    )

    before = _generate(model, tokenizer, _HELDOUT)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    final_loss = float("nan")
    for step in range(steps):
        ex = examples[step % len(examples)]
        enc = tokenizer(
            _format(tokenizer, ex),
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        )
        enc["labels"] = enc["input_ids"].clone()
        out = model(**enc)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        final_loss = float(loss.item())
        _log.info("step %d/%d loss=%.4f", step + 1, steps, final_loss)

    model.eval()
    after = _generate(model, tokenizer, _HELDOUT)

    out_dir = out_dir or (settings.cache_dir / "adapters" / method.value)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))

    return FineTuneResult(
        method=method,
        model_id=model_id,
        trainable_params=summary["trainable"],
        total_params=summary["total"],
        steps=steps,
        final_loss=final_loss,
        before=before,
        after=after,
        adapter_path=str(out_dir),
        note=note,
    )
