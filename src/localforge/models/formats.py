"""Detect model weight formats (docs/SPECIFICATION.md §3.1).

The lecture contrasts SafeTensors (mmap-friendly, zero-copy) with GGUF (the
quantized format Ollama/llama.cpp use) and legacy PyTorch ``.bin`` pickles.
localforge records which one a pulled model uses so the report can explain the
load path (e.g. AirLLM relies on SafeTensors + mmap).
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path

_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin", ".pt", ".pth")


class ModelFormat(StrEnum):
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    PYTORCH_BIN = "pytorch_bin"
    UNKNOWN = "unknown"


def detect_format(filenames: Iterable[str]) -> ModelFormat:
    """Classify a model from its file listing. SafeTensors wins when mixed."""
    names = [n.lower() for n in filenames]
    if any(n.endswith(".safetensors") for n in names):
        return ModelFormat.SAFETENSORS
    if any(n.endswith(".gguf") for n in names):
        return ModelFormat.GGUF
    if any(n.endswith((".bin", ".pt", ".pth")) for n in names):
        return ModelFormat.PYTORCH_BIN
    return ModelFormat.UNKNOWN


def weight_bytes(root: Path) -> int:
    """Total size in bytes of weight files under ``root`` (recursively)."""
    total = 0
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in _WEIGHT_SUFFIXES:
            total += path.stat().st_size
    return total
