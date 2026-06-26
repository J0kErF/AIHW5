"""API gatekeeper (docs guidelines §5.1).

A single chokepoint for outbound API calls (Hugging Face Hub downloads, the
Ollama HTTP endpoint). It centralizes two cross-cutting policies so individual
backends don't reimplement them:

- **rate limiting** — enforce a minimum interval between successive calls;
- **retry with backoff** — retry transient failures (network blips) a bounded
  number of times, while letting definitive errors (HTTP 404, auth) propagate.

    gk = ApiGatekeeper(RateLimitConfig(min_interval_s=0.0, max_retries=2))
    resp = gk.execute(urlopen, request, retry_on=(URLError,), no_retry=(HTTPError,))
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter, sleep
from typing import TypeVar

from localforge.core.logging import get_logger

_log = get_logger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class RateLimitConfig:
    min_interval_s: float = 0.0
    max_retries: int = 2
    backoff_s: float = 0.5


class ApiGatekeeper:
    """Centralized rate limiting + retry for outbound API calls."""

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        self._cfg = config or RateLimitConfig()
        self._last_call: float | None = None

    def _respect_rate_limit(self) -> None:
        if self._last_call is None or self._cfg.min_interval_s <= 0:
            return
        remaining = self._cfg.min_interval_s - (perf_counter() - self._last_call)
        if remaining > 0:
            sleep(remaining)

    def execute(
        self,
        fn: Callable[..., T],
        *args: object,
        retry_on: tuple[type[BaseException], ...] = (),
        no_retry: tuple[type[BaseException], ...] = (),
        **kwargs: object,
    ) -> T:
        """Call ``fn(*args, **kwargs)`` under the rate limit, retrying ``retry_on``.

        Exceptions matching ``no_retry`` (e.g. an ``HTTPError`` subclass of the
        retried ``URLError``) propagate immediately, as do exceptions not in
        ``retry_on``. After ``max_retries`` retries the last exception is re-raised.
        """
        self._respect_rate_limit()
        attempt = 0
        while True:
            try:
                result = fn(*args, **kwargs)
                self._last_call = perf_counter()
                return result
            except retry_on as exc:
                if no_retry and isinstance(exc, no_retry):
                    raise
                attempt += 1
                if attempt > self._cfg.max_retries:
                    raise
                _log.warning(
                    "transient API failure (%s); retry %d/%d",
                    type(exc).__name__,
                    attempt,
                    self._cfg.max_retries,
                )
                sleep(self._cfg.backoff_s * attempt)
