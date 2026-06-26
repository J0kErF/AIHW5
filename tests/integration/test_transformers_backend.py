"""Real-model test for the transformers backend.

Marked ``slow``: it downloads a small model on first run. Excluded from the
default/CI suite (run explicitly with ``-m slow``). The download-free coverage
of the run pipeline comes from FakeBackend.
"""

from __future__ import annotations

import pytest

from localforge.backends.runner import run_spec
from localforge.core.types import Backend, Dtype, RunSpec


@pytest.mark.slow
def test_transformers_cpu_greedy_generation() -> None:
    spec = RunSpec(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend=Backend.TRANSFORMERS,
        prompt="Reply with exactly one word: hello.",
        max_new_tokens=8,
        dtype=Dtype.FP32,
        device="cpu",
    )
    result = run_spec(spec)
    assert result.backend_available is True
    assert result.text.strip() != ""
    assert result.load_s > 0
    assert result.peak_ram_mb > 0
    assert result.peak_vram_mb is None  # CPU-only run


@pytest.mark.slow
def test_transformers_nf4_falls_back_on_cpu() -> None:
    spec = RunSpec(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend=Backend.TRANSFORMERS,
        prompt="Hi",
        max_new_tokens=4,
        dtype=Dtype.NF4,
        device="cpu",
    )
    result = run_spec(spec)
    # NF4 needs CUDA+bitsandbytes; on CPU it must still produce output with a note.
    assert result.backend_available is True
    assert result.text.strip() != ""
    assert result.note is not None and "fp32" in result.note
