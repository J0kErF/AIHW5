"""Tests for stable RunSpec hashing."""

from __future__ import annotations

from localforge.core.hashing import spec_hash
from localforge.core.types import Backend, Dtype, RunSpec


def _spec(**overrides: object) -> RunSpec:
    base: dict[str, object] = {
        "model_id": "m",
        "backend": Backend.TRANSFORMERS,
        "prompt": "p",
        "max_new_tokens": 8,
    }
    base.update(overrides)
    return RunSpec(**base)  # type: ignore[arg-type]


def test_hash_is_stable_for_equal_specs() -> None:
    assert spec_hash(_spec()) == spec_hash(_spec())


def test_hash_is_deterministic_length() -> None:
    h = spec_hash(_spec())
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_hash_changes_on_any_field() -> None:
    base = spec_hash(_spec())
    assert spec_hash(_spec(prompt="other")) != base
    assert spec_hash(_spec(dtype=Dtype.NF4)) != base
    assert spec_hash(_spec(seed=1)) != base
    assert spec_hash(_spec(max_new_tokens=9)) != base
    assert spec_hash(_spec(backend=Backend.AIRLLM)) != base
