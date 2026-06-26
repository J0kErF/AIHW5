"""Tests for the paging tracer, synthesis, persistence, replay, and hooks."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from localforge.core.types import PageAction, PageSource
from localforge.paging.airllm_hook import instrument_layers, synthesize_airllm_trace
from localforge.paging.events import MemorySink, read_stream, write_stream
from localforge.paging.replay import load_and_replay, summarize
from localforge.paging.tracer import PagingTracer

_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "sample_paging.jsonl"


def test_tracer_fans_out_to_sinks() -> None:
    tracer = PagingTracer()
    sink = MemorySink()
    tracer.subscribe(sink)
    tracer.start()
    tracer.emit(3, PageAction.FAULT, 1024, PageSource.MMAP)
    assert len(sink.events) == 1
    assert sink.events[0].layer == 3
    assert sink.events[0].action is PageAction.FAULT


def test_synthesize_has_faults_and_evicts() -> None:
    events = synthesize_airllm_trace(n_layers=16, layer_mb=300, ram_ceiling_mb=1500, passes=2)
    summary = summarize(events)
    assert summary.n_faults > 0
    assert summary.n_evicts > 0
    # capacity = 1500 // 300 = 5 resident blocks max.
    assert summary.peak_resident_layers == 5
    assert summary.distinct_layers == 16


def test_stream_round_trip(tmp_path: Path) -> None:
    events = synthesize_airllm_trace(n_layers=8, layer_mb=200, ram_ceiling_mb=800, passes=1)
    path = write_stream(events, tmp_path / "s.jsonl")
    restored = read_stream(path)
    assert restored == events


def test_replay_fixture() -> None:
    tracer = PagingTracer()
    sink = MemorySink()
    tracer.subscribe(sink)
    events = load_and_replay(_FIXTURE, tracer)
    assert len(events) == len(sink.events) > 0
    summary = summarize(events)
    assert summary.n_faults > summary.n_hits or summary.n_faults > 0


def test_instrument_layers_emits_per_block() -> None:
    class Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Inner()

    model = FakeModel()
    tracer = PagingTracer()
    sink = MemorySink()
    tracer.subscribe(sink)
    tracer.start()

    n = instrument_layers(model, tracer)
    assert n == 3

    x = torch.randn(1, 4)
    for block in model.model.layers:
        x = block(x)
    assert len(sink.events) == 3
    assert {e.layer for e in sink.events} == {0, 1, 2}
    assert all(e.action is PageAction.FAULT for e in sink.events)


def test_instrument_unknown_architecture_returns_zero() -> None:
    tracer = PagingTracer()
    assert instrument_layers(nn.Linear(4, 4), tracer) == 0
