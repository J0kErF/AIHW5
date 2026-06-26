"""Suite definitions: expand a YAML matrix into RunSpecs (docs/IMPLEMENTATION.md §2.5).

A suite is the cartesian product of models x backends x dtypes sharing one
prompt. Because Ollama addresses models by its own tags (e.g. ``qwen2.5:0.5b``)
rather than Hugging Face repo ids, a suite may map HF ids to Ollama tags via
``ollama_aliases`` so the "same" model can be compared across backends.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from localforge.core.errors import ConfigError
from localforge.core.types import Backend, Dtype, RunSpec


class SuiteDoc(BaseModel):
    suite_id: str
    models: list[str]
    backends: list[Backend]
    dtypes: list[Dtype]
    prompt: str
    max_new_tokens: int
    seed: int = 0
    ollama_aliases: dict[str, str] = {}


def load_suite(path: Path) -> SuiteDoc:
    """Parse and validate a suite YAML file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read suite {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"suite {path} must be a YAML mapping")
    try:
        return SuiteDoc.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid suite {path}:\n{exc}") from exc


def expand_suite(doc: SuiteDoc) -> list[RunSpec]:
    """Expand the suite into one RunSpec per (model, backend, dtype) cell."""
    specs: list[RunSpec] = []
    for model in doc.models:
        for backend in doc.backends:
            model_id = model
            if backend is Backend.OLLAMA and model in doc.ollama_aliases:
                model_id = doc.ollama_aliases[model]
            for dtype in doc.dtypes:
                specs.append(
                    RunSpec(
                        model_id=model_id,
                        backend=backend,
                        prompt=doc.prompt,
                        max_new_tokens=doc.max_new_tokens,
                        dtype=dtype,
                        seed=doc.seed,
                    )
                )
    return specs
