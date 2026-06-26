"""A tiny local registry of pulled models (docs/SPECIFICATION.md §3.1).

Records what was downloaded (id, format, size, local path) as a single JSON file
under the cache dir, so subsequent runs can resolve a model without re-pulling.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from localforge.models.formats import ModelFormat


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    format: ModelFormat
    size_bytes: int
    path: str

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


class ModelRegistry:
    def __init__(self, cache_dir: Path) -> None:
        self._path = cache_dir / "registry.json"

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        data: dict[str, dict[str, Any]] = json.loads(self._path.read_text(encoding="utf-8"))
        return data

    def record(self, info: ModelInfo) -> None:
        data = self._load()
        data[info.model_id] = asdict(info)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def get(self, model_id: str) -> ModelInfo | None:
        entry = self._load().get(model_id)
        if entry is None:
            return None
        return ModelInfo(
            model_id=str(entry["model_id"]),
            format=ModelFormat(str(entry["format"])),
            size_bytes=int(entry["size_bytes"]),
            path=str(entry["path"]),
        )

    def all(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                model_id=str(e["model_id"]),
                format=ModelFormat(str(e["format"])),
                size_bytes=int(e["size_bytes"]),
                path=str(e["path"]),
            )
            for e in self._load().values()
        ]
