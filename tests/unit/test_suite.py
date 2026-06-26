"""Tests for suite loading/expansion."""

from __future__ import annotations

from pathlib import Path

import pytest

from localforge.config.suite import SuiteDoc, expand_suite, load_suite
from localforge.core.errors import ConfigError
from localforge.core.types import Backend, Dtype

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_expand_cartesian_product() -> None:
    doc = SuiteDoc(
        suite_id="t",
        models=["m1", "m2"],
        backends=[Backend.TRANSFORMERS, Backend.OLLAMA],
        dtypes=[Dtype.FP32, Dtype.NF4],
        prompt="p",
        max_new_tokens=8,
    )
    specs = expand_suite(doc)
    assert len(specs) == 2 * 2 * 2
    assert {s.model_id for s in specs} == {"m1", "m2"}


def test_ollama_alias_applied_only_to_ollama() -> None:
    doc = SuiteDoc(
        suite_id="t",
        models=["Qwen/Qwen2.5-0.5B-Instruct"],
        backends=[Backend.TRANSFORMERS, Backend.OLLAMA],
        dtypes=[Dtype.FP32],
        prompt="p",
        max_new_tokens=8,
        ollama_aliases={"Qwen/Qwen2.5-0.5B-Instruct": "qwen2.5:0.5b"},
    )
    by_backend = {s.backend: s.model_id for s in expand_suite(doc)}
    assert by_backend[Backend.OLLAMA] == "qwen2.5:0.5b"
    assert by_backend[Backend.TRANSFORMERS] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_load_demo_suite_file() -> None:
    doc = load_suite(_PROJECT_ROOT / "config" / "suites" / "demo.yaml")
    assert doc.suite_id == "demo"
    assert Backend.AIRLLM in doc.backends
    assert expand_suite(doc)


def test_bad_suite_raises_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("models: [m]\nbackends: [transformers]\n", encoding="utf-8")  # missing fields
    with pytest.raises(ConfigError):
        load_suite(bad)


def test_non_mapping_suite_raises(tmp_path: Path) -> None:
    bad = tmp_path / "list.yaml"
    bad.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_suite(bad)
