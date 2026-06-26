"""Tests for the ApiGatekeeper (rate limiting + retry)."""

from __future__ import annotations

import pytest

import localforge.core.gatekeeper as gk_mod
from localforge.core.gatekeeper import ApiGatekeeper, RateLimitConfig


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record sleeps instead of actually sleeping."""
    slept: list[float] = []
    monkeypatch.setattr(gk_mod, "sleep", lambda s: slept.append(s))
    return slept


def test_execute_returns_result() -> None:
    gk = ApiGatekeeper()
    assert gk.execute(lambda x: x + 1, 41) == 42


def test_execute_passes_kwargs() -> None:
    gk = ApiGatekeeper()
    assert gk.execute(lambda *, a, b: a * b, a=6, b=7) == 42


def test_non_retryable_exception_propagates_immediately() -> None:
    gk = ApiGatekeeper(RateLimitConfig(max_retries=3))
    calls = {"n": 0}

    def boom() -> None:
        calls["n"] += 1
        raise ValueError("definitive")

    with pytest.raises(ValueError):
        gk.execute(boom, retry_on=(KeyError,))
    assert calls["n"] == 1  # not retried


def test_retries_then_succeeds(_no_sleep: list[float]) -> None:
    gk = ApiGatekeeper(RateLimitConfig(max_retries=3, backoff_s=0.5))
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("blip")
        return "ok"

    assert gk.execute(flaky, retry_on=(ConnectionError,)) == "ok"
    assert calls["n"] == 3
    assert _no_sleep == [0.5, 1.0]  # backoff * attempt for the two retries


def test_raises_after_max_retries() -> None:
    gk = ApiGatekeeper(RateLimitConfig(max_retries=2))
    calls = {"n": 0}

    def always_fail() -> None:
        calls["n"] += 1
        raise TimeoutError("down")

    with pytest.raises(TimeoutError):
        gk.execute(always_fail, retry_on=(TimeoutError,))
    assert calls["n"] == 3  # initial + 2 retries


def test_rate_limit_waits_between_calls(
    _no_sleep: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = {"t": 0.0}
    monkeypatch.setattr(gk_mod, "perf_counter", lambda: clock["t"])
    gk = ApiGatekeeper(RateLimitConfig(min_interval_s=1.0))
    gk.execute(lambda: None)  # first call sets last_call (t=0)
    clock["t"] = 0.3  # only 0.3s elapsed before the next call
    gk.execute(lambda: None)
    assert _no_sleep and abs(_no_sleep[-1] - 0.7) < 1e-9  # waited the remaining 0.7s
