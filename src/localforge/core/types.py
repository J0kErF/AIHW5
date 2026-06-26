"""Core domain types shared across localforge.

These are the typed values that flow through the whole pipeline:
``RunSpec`` describes a job, ``RunResult`` records its measured outcome, and
``PagingEvent`` is one unit of the AirLLM paging trace. See
docs/SPECIFICATION.md §5 and docs/IMPLEMENTATION.md §2.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Backend(StrEnum):
    """Inference engine selector."""

    TRANSFORMERS = "transformers"
    OLLAMA = "ollama"
    AIRLLM = "airllm"
    FAKE = "fake"  # deterministic test backend (see tests/conftest.py)


class Dtype(StrEnum):
    """Requested weight precision. ``NF4`` is a CUDA-only path (see DECISIONS.md D1)."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    NF4 = "nf4"


class Device(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class PageAction(StrEnum):
    """What happened to a model layer during AirLLM streaming."""

    FAULT = "fault"  # layer not resident -> loaded from disk
    HIT = "hit"  # layer already resident -> reused
    EVICT = "evict"  # layer dropped from RAM


class PageSource(StrEnum):
    MMAP = "mmap"  # zero-copy via the OS page cache
    DISK = "disk"  # explicit read


class RunSpec(BaseModel):
    """Immutable description of a single inference job.

    Frozen so it can be hashed into a stable ``spec_hash`` and used as a cache /
    result key (see :mod:`localforge.core.hashing`).
    """

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(min_length=1, description="HF model id or Ollama tag")
    backend: Backend
    prompt: str
    max_new_tokens: int = Field(ge=1, le=4096)
    dtype: Dtype = Dtype.FP32
    seed: int = 0
    device: Device = Device.AUTO


class RunResult(BaseModel):
    """Measured outcome of one run.

    A run that could not execute because its backend was unavailable is *not* an
    error: it is recorded with ``backend_available=False`` and a ``note`` so a
    comparison suite always completes (see docs/SPECIFICATION.md §3.4.1).
    """

    spec_hash: str
    backend: Backend
    model_id: str
    text: str = ""
    load_s: float = 0.0
    prefill_ms: float = 0.0
    decode_tok_s: float | None = None
    peak_ram_mb: float = 0.0
    peak_vram_mb: float | None = None
    backend_available: bool = True
    note: str | None = None

    @classmethod
    def skipped(cls, spec: RunSpec, spec_hash: str, reason: str) -> RunResult:
        """Build a non-fatal 'backend unavailable' result for ``spec``."""
        return cls(
            spec_hash=spec_hash,
            backend=spec.backend,
            model_id=spec.model_id,
            backend_available=False,
            note=reason,
        )


class PagingEvent(BaseModel):
    """One instrumented event from AirLLM layer streaming (docs/SPECIFICATION.md §3.6)."""

    layer: int = Field(ge=0)
    action: PageAction
    bytes: int = Field(ge=0)
    source: PageSource
    t_ms: float = Field(ge=0.0)
