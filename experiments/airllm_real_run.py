"""Run a REAL AirLLM layer-streamed pass on CPU and print TTFT/TPOT/RAM (spec §5.3).

AirLLM 2.11 is incompatible with the transformers 5.x that localforge's main env
uses (DynamicCache is not subscriptable; the Qwen2 layer forward now requires a
``position_embeddings`` argument AirLLM's hand-rolled loop never passes). It runs
on **transformers 4.41**, so this script is meant for a dedicated env:

    uv venv .airllm-env --python 3.12
    uv pip install --python .airllm-env/Scripts/python.exe \\
        torch==2.4.1 transformers==4.41.2 "optimum<2" airllm sentencepiece accelerate
    .airllm-env/Scripts/python.exe experiments/airllm_real_run.py

It applies the documented compatibility fixes (see docs/RE_AIRLLM.md):
  1) shim optimum.bettertransformer (removed in optimum>=2),
  2) no-op torch.cuda.* (a CPU-only torch raises on empty_cache),
  3) shard the model (AirLLM needs a safetensors index),
  4) untie lm_head (Qwen2.5-0.5B ties it to embed_tokens; the splitter needs it),
  5) device="cpu", dtype=float32, use_cache=False.

Recorded result (Qwen2.5-0.5B, 4-core CPU): TTFT ~4.1 s, TPOT ~3.8 s/token,
peak RSS ~370 MB — vs the transformers backend's ~3132 MB and 0.084 s/token.
"""

from __future__ import annotations

import sys
import tempfile
import time
import types

import psutil
import torch
from torch import nn


def _apply_compat_shims() -> None:
    bt = types.ModuleType("optimum.bettertransformer")
    bt.BetterTransformer = type(  # type: ignore[attr-defined]
        "BetterTransformer", (), {"transform": staticmethod(lambda m, **k: m)}
    )
    sys.modules["optimum.bettertransformer"] = bt
    torch.cuda.empty_cache = lambda: None  # type: ignore[assignment]
    torch.cuda.reset_peak_memory_stats = lambda *a, **k: None  # type: ignore[assignment]
    torch.cuda.synchronize = lambda *a, **k: None  # type: ignore[assignment]


def main(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
    _apply_compat_shims()
    from airllm import AutoModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = tempfile.mkdtemp(prefix="airllm_real_")
    base = AutoModelForCausalLM.from_pretrained(model_id)
    base.config.tie_word_embeddings = False
    base.lm_head.weight = nn.Parameter(base.get_input_embeddings().weight.detach().clone())
    base.save_pretrained(out_dir, max_shard_size="100MB")
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out_dir)
    del base

    proc = psutil.Process()
    model = AutoModel.from_pretrained(out_dir, device="cpu", dtype=torch.float32)
    tokens = model.tokenizer(
        ["Explain virtual memory in one sentence."],
        return_tensors="pt",
        return_attention_mask=False,
    )

    t0 = time.time()
    model.generate(tokens["input_ids"], max_new_tokens=1, use_cache=False, return_dict_in_generate=True)
    ttft = time.time() - t0

    t0 = time.time()
    out = model.generate(tokens["input_ids"], max_new_tokens=3, use_cache=False, return_dict_in_generate=True)
    total3 = time.time() - t0

    peak_mb = proc.memory_info().rss / 1e6
    tpot = (total3 - ttft) / 2 if total3 > ttft else total3 / 3
    seq = getattr(out, "sequences", out)
    text = model.tokenizer.decode(seq[0], skip_special_tokens=True)
    print(f"TTFT={ttft:.1f}s  TPOT={tpot:.1f}s/token  peak_rss={peak_mb:.0f}MB")
    print(f"text={text!r}")


if __name__ == "__main__":
    main()
