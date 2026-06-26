"""Shared pytest fixtures.

FakeBackend (a deterministic, dependency-free engine) lives in
``localforge.backends.fake_backend`` so the whole run -> compare -> report
pipeline is exercisable with no model downloads and no GPU.
"""

from __future__ import annotations

import pytest

from localforge.core.types import Backend, RunSpec


@pytest.fixture()
def fake_spec() -> RunSpec:
    return RunSpec(
        model_id="fake/tiny",
        backend=Backend.FAKE,
        prompt="hello",
        max_new_tokens=8,
    )
