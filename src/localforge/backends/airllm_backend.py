"""AirLLM backend: layer-by-layer streaming inference (docs/SPECIFICATION.md §3.6).

AirLLM runs a model one transformer block at a time, loading each block's weights
from disk (SafeTensors + mmap), computing, then releasing it — so a model far
larger than RAM can run within a bounded working set. This is the OS virtual-
memory analogy the project visualizes.

``airllm`` is an optional dependency (extra: ``airllm``). When absent, this
backend reports unavailable and the runner skips it — no crash. The real run and
the per-layer paging instrumentation are wired in Phase 5 (T9a/T15); the
``tracer`` hook point is reserved here.

Note: AirLLM's ``generate`` returns the full completion rather than a token
stream, so for this backend the profiler records generation as a single prefill
window (decode_tok_s is reported as n/a). For AirLLM the meaningful signal is the
per-layer paging trace, not decode throughput.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from localforge.backends.base import register
from localforge.config.settings import load_settings
from localforge.core.capabilities import probe_airllm
from localforge.core.types import RunSpec


@register
class AirLLMBackend:
    name = "airllm"

    def __init__(self) -> None:
        self.note: str | None = None
        self._model: Any = None
        self.tracer: Any = None  # set by the paging visualizer (T15)
        self._ceiling_mb = load_settings().airllm_ram_ceiling_mb

    def is_available(self) -> tuple[bool, str]:
        return probe_airllm()

    def load(self, spec: RunSpec) -> None:
        from airllm import AutoModel  # lazy import: optional dependency

        # AirLLM streams layers from the HF cache; compression keeps the per-layer
        # working set small enough for a CPU-only box.
        self._model = AutoModel.from_pretrained(spec.model_id)
        self.note = f"AirLLM layer streaming (working-set target ~{self._ceiling_mb} MB)"

    def generate(self, spec: RunSpec) -> Iterator[str]:
        tokenizer = self._model.tokenizer
        inputs = tokenizer([spec.prompt], return_tensors="pt", return_attention_mask=False)
        output = self._model.generate(
            inputs["input_ids"],
            max_new_tokens=spec.max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
        )
        sequences = getattr(output, "sequences", output)
        text = tokenizer.decode(sequences[0], skip_special_tokens=True)
        # Single yield: first (and only) piece marks end of the prefill window.
        yield text
