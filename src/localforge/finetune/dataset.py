"""Tiny instruction (SFT) dataset loader (docs/SPECIFICATION.md §3.5).

Reads a JSONL file of ``{"instruction", "response"}`` records — small by design,
so a LoRA adapter can be trained end-to-end on CPU in a handful of steps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from localforge.core.errors import ConfigError


@dataclass(frozen=True)
class Example:
    instruction: str
    response: str


def load_sft(path: Path) -> list[Example]:
    if not path.exists():
        raise ConfigError(f"dataset not found: {path}")
    examples: list[Example] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            examples.append(Example(instruction=obj["instruction"], response=obj["response"]))
        except (json.JSONDecodeError, KeyError) as exc:
            raise ConfigError(f"{path}:{i}: bad SFT record ({exc})") from exc
    if not examples:
        raise ConfigError(f"dataset is empty: {path}")
    return examples
