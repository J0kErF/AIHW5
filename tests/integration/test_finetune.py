"""Fine-tuning tests: offline dataset checks + a slow real CPU LoRA train."""

from __future__ import annotations

from pathlib import Path

import pytest

from localforge.config.settings import load_settings
from localforge.core.errors import ConfigError
from localforge.finetune.adapters import FineTuneMethod
from localforge.finetune.dataset import load_sft

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA = _PROJECT_ROOT / "data" / "finetune" / "tiny_sft.jsonl"


def test_load_sft_reads_examples() -> None:
    examples = load_sft(_DATA)
    assert len(examples) >= 4
    assert all(e.instruction and e.response for e in examples)


def test_load_sft_rejects_bad_record(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"instruction": "x"}\n', encoding="utf-8")  # missing response
    with pytest.raises(ConfigError):
        load_sft(bad)


def test_load_sft_rejects_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_sft(empty)


@pytest.mark.slow
def test_real_lora_train_cpu(tmp_path: Path) -> None:
    from localforge.finetune.trainer import train_adapter

    result = train_adapter(
        "Qwen/Qwen2.5-0.5B-Instruct",
        FineTuneMethod.LORA,
        _DATA,
        load_settings(),
        steps=6,
        out_dir=tmp_path / "adapter",
    )
    # Only the low-rank A/B matrices are trainable -> a tiny fraction of the model.
    assert 0 < result.trainable_params < result.total_params
    assert result.trainable_pct < 5.0
    assert isinstance(result.before, str)
    assert isinstance(result.after, str)
    assert (tmp_path / "adapter").exists()
    assert result.final_loss == result.final_loss  # not NaN
