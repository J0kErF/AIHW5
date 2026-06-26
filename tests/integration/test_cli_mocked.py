"""CLI tests for command paths that wrap heavy work (mocked)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import localforge.finetune.trainer as trainer_mod
import localforge.models.acquire as acquire_mod
from localforge.cli.app import app
from localforge.finetune.adapters import FineTuneMethod
from localforge.finetune.trainer import FineTuneResult
from localforge.models.formats import ModelFormat
from localforge.models.registry import ModelInfo

runner = CliRunner()


def test_baseline_command_writes_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["baseline", "--results-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "baseline").exists()


def test_pull_command_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    info = ModelInfo("org/model", ModelFormat.SAFETENSORS, 2048, "/cache/org/model")
    monkeypatch.setattr(acquire_mod, "pull_model", lambda *a, **k: info)
    result = runner.invoke(app, ["pull", "org/model"])
    assert result.exit_code == 0
    assert "org/model" in result.output


def test_finetune_command_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_result = FineTuneResult(
        method=FineTuneMethod.LORA,
        model_id="m",
        trainable_params=1000,
        total_params=500000,
        steps=4,
        final_loss=1.23,
        before="b",
        after="a",
        adapter_path="/tmp/adapter",
        note=None,
    )
    monkeypatch.setattr(trainer_mod, "train_adapter", lambda *a, **k: fake_result)
    result = runner.invoke(
        app,
        ["finetune", "--method", "lora", "--steps", "4", "--data", "data/finetune/tiny_sft.jsonl"],
    )
    assert result.exit_code == 0, result.output
    assert "before:" in result.output
    assert "after" in result.output


def test_finetune_rejects_bad_method() -> None:
    result = runner.invoke(app, ["finetune", "--method", "nonsense"])
    assert result.exit_code != 0
