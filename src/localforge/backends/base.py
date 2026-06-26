"""Backend abstraction: Protocol, registry/factory, and the Null-object fallback.

Backends are interchangeable inference engines selected by name (Strategy +
Registry Factory, docs/IMPLEMENTATION.md §2.1). Engine modules import their heavy
dependencies *lazily inside methods*, so importing a backend module is cheap and
a missing optional dependency surfaces through ``is_available()`` rather than an
import crash (docs/DECISIONS.md D1).
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from localforge.core.errors import BackendUnavailable, UnknownBackend
from localforge.core.types import RunSpec


@runtime_checkable
class InferenceBackend(Protocol):
    name: str

    def is_available(self) -> tuple[bool, str]:
        """Return ``(ok, reason)``. Never raises."""
        ...

    def load(self, spec: RunSpec) -> None:
        """Load weights for ``spec``. Called once before :meth:`generate`."""
        ...

    def generate(self, spec: RunSpec) -> Iterator[str]:
        """Yield generated text pieces; the first yield marks end of prefill."""
        ...


_REGISTRY: dict[str, type[InferenceBackend]] = {}

# Lazy import targets: a backend name -> the module that registers it.
_MODULES: dict[str, str] = {
    "fake": "localforge.backends.fake_backend",
    "transformers": "localforge.backends.transformers_backend",
    "ollama": "localforge.backends.ollama_backend",
    "airllm": "localforge.backends.airllm_backend",
}


def register(cls: type[InferenceBackend]) -> type[InferenceBackend]:
    """Class decorator: register a backend under its ``name``."""
    _REGISTRY[cls.name] = cls
    return cls


def make_backend(name: str) -> InferenceBackend:
    """Resolve a backend by name, importing its module on demand.

    Unknown names raise :class:`UnknownBackend`. A module that fails to import
    (e.g. an optional engine whose top-level import is broken) degrades to an
    :class:`UnavailableBackend` rather than crashing the caller.
    """
    if name not in _REGISTRY:
        module = _MODULES.get(name)
        if module is None:
            raise UnknownBackend(name)
        try:
            importlib.import_module(module)
        except ImportError as exc:
            return UnavailableBackend(name, f"import failed: {exc}")
    if name not in _REGISTRY:
        raise UnknownBackend(name)
    return _REGISTRY[name]()


def available_backends() -> tuple[str, ...]:
    return tuple(_MODULES)


class UnavailableBackend:
    """Null-object backend: reports unavailable and refuses to run."""

    name = "unavailable"

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason

    def is_available(self) -> tuple[bool, str]:
        return (False, self.reason)

    def load(self, spec: RunSpec) -> None:
        raise BackendUnavailable(self.name, self.reason)

    def generate(self, spec: RunSpec) -> Iterator[str]:
        raise BackendUnavailable(self.name, self.reason)
        yield  # pragma: no cover - makes this a generator function
