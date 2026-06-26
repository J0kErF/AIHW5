"""Smoke test: the package imports and exposes a version. No heavy deps."""

from __future__ import annotations

import localforge


def test_version_is_exposed() -> None:
    assert isinstance(localforge.__version__, str)
    assert localforge.__version__.count(".") >= 2
