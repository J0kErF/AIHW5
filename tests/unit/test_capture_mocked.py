"""Unit test for the per-block trace capture (transformers mocked)."""

from __future__ import annotations

import pytest

from localforge.core.types import PageAction
from localforge.viz.capture import capture_transformers_trace
from tests._fakes import install_fake_transformers


def test_capture_emits_one_event_per_block(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_transformers(monkeypatch, n_layers=3)
    events = capture_transformers_trace("fake/model", "explain paging", max_new_tokens=2)
    # 3 blocks executed once by the fake generate -> 3 fault events.
    assert len(events) == 3
    assert {e.layer for e in events} == {0, 1, 2}
    assert all(e.action is PageAction.FAULT for e in events)
