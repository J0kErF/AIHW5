"""A deterministic, dependency-free backend for tests and smoke runs.

FakeBackend makes the whole run -> compare -> report pipeline exercisable with
no model downloads and no GPU (docs/IMPLEMENTATION.md §9.2). Output is a fixed,
seed-independent token stream so assertions are stable.
"""

from __future__ import annotations

from collections.abc import Iterator

from localforge.backends.base import register
from localforge.core.types import RunSpec

_WORDS = ["virtual", "memory", "maps", "disk", "to", "ram", "on", "demand"]


@register
class FakeBackend:
    name = "fake"

    def is_available(self) -> tuple[bool, str]:
        return (True, "fake backend always available")

    def load(self, spec: RunSpec) -> None:
        # No weights to load; exists so the runner's load timing has something to wrap.
        self._loaded = True

    def generate(self, spec: RunSpec) -> Iterator[str]:
        n = min(spec.max_new_tokens, len(_WORDS))
        for word in _WORDS[:n]:
            yield word + " "
