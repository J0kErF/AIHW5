"""Runtime capability detection.

Every probe is defensive: it returns ``(available, reason)`` and never raises,
so localforge stays usable on a CPU-only machine with no GPU, no Ollama daemon,
and AirLLM/bitsandbytes not installed (docs/DECISIONS.md D1).
"""

from __future__ import annotations

import importlib.util
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

Probe = tuple[bool, str]


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def probe_cuda() -> Probe:
    """True if a CUDA device is visible to torch."""
    if not _has_module("torch"):
        return False, "torch not installed"
    try:
        import torch

        if torch.cuda.is_available():
            return True, f"{torch.cuda.device_count()} CUDA device(s)"
        return False, "no CUDA device visible to torch"
    except Exception as exc:  # torch can raise on broken driver installs
        return False, f"torch CUDA check failed: {exc}"


def probe_bitsandbytes() -> Probe:
    """bitsandbytes (NF4/QLoRA) requires CUDA; report both conditions."""
    if not _has_module("bitsandbytes"):
        return False, "bitsandbytes not installed (extra: gpu)"
    cuda_ok, cuda_reason = probe_cuda()
    if not cuda_ok:
        return False, f"bitsandbytes present but {cuda_reason}"
    return True, "bitsandbytes + CUDA available"


def probe_airllm() -> Probe:
    if _has_module("airllm"):
        return True, "airllm installed"
    return False, "airllm not installed (extra: airllm)"


def probe_ollama(base_url: str = "http://localhost:11434/v1") -> Probe:
    """True if an Ollama daemon answers on the configured host:port."""
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True, f"ollama daemon reachable at {host}:{port}"
    except OSError:
        return False, f"no ollama daemon at {host}:{port} (run `ollama serve`)"


@dataclass(frozen=True)
class Capabilities:
    cuda: Probe
    bitsandbytes: Probe
    airllm: Probe
    ollama: Probe

    def as_dict(self) -> dict[str, dict[str, object]]:
        return {
            name: {"available": probe[0], "reason": probe[1]}
            for name, probe in (
                ("cuda", self.cuda),
                ("bitsandbytes", self.bitsandbytes),
                ("airllm", self.airllm),
                ("ollama", self.ollama),
            )
        }


def probe_capabilities(ollama_base_url: str = "http://localhost:11434/v1") -> Capabilities:
    """Probe all engines. Pure, side-effect-free, never raises."""
    return Capabilities(
        cuda=probe_cuda(),
        bitsandbytes=probe_bitsandbytes(),
        airllm=probe_airllm(),
        ollama=probe_ollama(ollama_base_url),
    )
