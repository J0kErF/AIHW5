"""Tests for the filesystem ResultStore."""

from __future__ import annotations

from pathlib import Path

from localforge.core.types import Backend, RunResult
from localforge.reporting.store import ResultStore


def _result(spec_hash: str, available: bool = True) -> RunResult:
    return RunResult(
        spec_hash=spec_hash,
        backend=Backend.TRANSFORMERS,
        model_id="m",
        text="hi",
        load_s=1.0,
        prefill_ms=10.0,
        decode_tok_s=5.0,
        peak_ram_mb=100.0,
        backend_available=available,
    )


def test_save_and_load_result(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    store.save_result(_result("abc"))
    loaded = store.load_result("abc")
    assert loaded is not None
    assert loaded.spec_hash == "abc"
    assert loaded.text == "hi"
    assert store.load_result("missing") is None


def test_save_suite_records_metadata_and_hashes(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    results = [_result("a"), _result("b", available=False)]
    store.save_suite("demo", results)

    run = store.load_suite("demo")
    assert run is not None
    assert run.suite_id == "demo"
    assert set(run.spec_hashes) == {"a", "b"}
    assert run.host
    assert run.created_utc

    loaded = store.load_suite_results("demo")
    assert {r.spec_hash for r in loaded} == {"a", "b"}
