"""Tests for core domain types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from localforge.core.types import (
    Backend,
    Dtype,
    PageAction,
    PageSource,
    PagingEvent,
    RunResult,
    RunSpec,
)


def _spec(**overrides: object) -> RunSpec:
    base: dict[str, object] = {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "backend": Backend.TRANSFORMERS,
        "prompt": "hello",
        "max_new_tokens": 16,
    }
    base.update(overrides)
    return RunSpec(**base)  # type: ignore[arg-type]


def test_runspec_is_frozen() -> None:
    spec = _spec()
    with pytest.raises(ValidationError):
        spec.prompt = "changed"  # type: ignore[misc]


def test_runspec_defaults() -> None:
    spec = _spec()
    assert spec.dtype is Dtype.FP32
    assert spec.seed == 0


def test_runspec_validates_bounds() -> None:
    with pytest.raises(ValidationError):
        _spec(max_new_tokens=0)
    with pytest.raises(ValidationError):
        _spec(model_id="")


def test_runspec_json_round_trip() -> None:
    spec = _spec(dtype=Dtype.NF4, seed=7)
    restored = RunSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec


def test_runresult_skipped_factory() -> None:
    spec = _spec(backend=Backend.OLLAMA)
    result = RunResult.skipped(spec, "abc123", "daemon down")
    assert result.backend_available is False
    assert result.note == "daemon down"
    assert result.backend is Backend.OLLAMA
    assert result.peak_vram_mb is None


def test_paging_event_round_trip() -> None:
    ev = PagingEvent(layer=3, action=PageAction.FAULT, bytes=1024, source=PageSource.MMAP, t_ms=1.5)
    restored = PagingEvent.model_validate_json(ev.model_dump_json())
    assert restored == ev
    with pytest.raises(ValidationError):
        PagingEvent(layer=-1, action=PageAction.HIT, bytes=0, source=PageSource.DISK, t_ms=0.0)
