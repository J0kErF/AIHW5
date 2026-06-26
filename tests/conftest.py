"""Shared pytest fixtures.

The FakeBackend and recorded paging fixtures (which let the full
run -> compare -> report -> visualize pipeline run with no model downloads and
no GPU) are added in Phase 2 (T5) and Phase 5 (T15). This stub keeps the test
package importable from Task 1 onward.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def anyio_backend() -> str:  # placeholder fixture; real fixtures arrive in T5
    return "asyncio"
