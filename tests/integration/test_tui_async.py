"""Headless test of the Textual paging TUI."""

from __future__ import annotations

import asyncio

from localforge.paging.airllm_hook import synthesize_airllm_trace
from localforge.viz.tui import PagingApp


def test_paging_app_ingests_events_headless() -> None:
    events = synthesize_airllm_trace(n_layers=8, layer_mb=200, ram_ceiling_mb=800, passes=1)
    app = PagingApp(events, 8, interval=0.001)

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            await pilot.pause(0.05)

    asyncio.run(_run())
    assert app._i > 0  # the timer advanced through the stream
