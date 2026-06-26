"""Exception hierarchy for localforge (docs/IMPLEMENTATION.md §7.1).

Only ``ConfigError`` and ``AuthError`` are fatal at the CLI boundary.
``BackendUnavailable`` is caught by the runner and turned into a non-fatal
skipped :class:`~localforge.core.types.RunResult`.
"""

from __future__ import annotations


class LocalforgeError(Exception):
    """Base class for all localforge errors."""


class ConfigError(LocalforgeError):
    """Invalid configuration (bad suite YAML, missing settings). Fatal, exit 2."""


class AuthError(LocalforgeError):
    """Missing/invalid credentials, e.g. no Hugging Face token. Fatal."""


class UnknownBackend(LocalforgeError):
    """Requested a backend name that is not registered."""

    def __init__(self, name: str) -> None:
        super().__init__(f"unknown backend: {name!r}")
        self.name = name


class BackendUnavailable(LocalforgeError):
    """A known backend cannot run here (daemon down, package absent). Non-fatal."""

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"backend {name!r} unavailable: {reason}")
        self.name = name
        self.reason = reason


class ResourceError(LocalforgeError):
    """Ran out of a hardware resource (e.g. RAM during layer streaming)."""
