"""Tokenization + generation helpers for the trainer (docs/SPECIFICATION.md §3.5).

Kept separate from the training loop so each module stays small and focused.
"""

from __future__ import annotations

from typing import Any

from localforge.finetune.dataset import Example


def format_example(tokenizer: Any, ex: Example) -> str:
    """Render one SFT example as a single training string."""
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "user", "content": ex.instruction},
            {"role": "assistant", "content": ex.response},
        ]
        return str(tokenizer.apply_chat_template(messages, tokenize=False))
    return f"{ex.instruction}\n{ex.response}"


def generate_reply(model: Any, tokenizer: Any, prompt: str, max_new: int = 16) -> str:
    """Greedy short generation for the held-out before/after comparison."""
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
