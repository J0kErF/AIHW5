"""Ollama backend over its OpenAI-compatible API (docs/SPECIFICATION.md §3.2.1).

The lecture (L08 §6) serves models through Ollama's OpenAI-compatible endpoint at
``http://localhost:11434/v1``. We talk to it with the stdlib only (no SDK), which
keeps the backend dependency-free and makes the wire protocol explicit for the
reverse-engineering write-up (docs/RE_OLLAMA.md).

``model_id`` here is an Ollama tag (e.g. ``qwen2.5:0.5b``), not an HF repo id.
A missing daemon or un-pulled model degrades to a skipped result, never a crash.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator

from localforge.backends.base import register
from localforge.config.settings import load_settings
from localforge.core.capabilities import probe_ollama
from localforge.core.errors import BackendUnavailable
from localforge.core.gatekeeper import ApiGatekeeper, RateLimitConfig
from localforge.core.types import RunSpec


@register
class OllamaBackend:
    name = "ollama"

    def __init__(self) -> None:
        self.note: str | None = None
        self._base_url = load_settings().ollama_base_url.rstrip("/")
        # Centralized outbound-call policy: retry transient network blips.
        self._gatekeeper = ApiGatekeeper(RateLimitConfig(min_interval_s=0.0, max_retries=2))

    def is_available(self) -> tuple[bool, str]:
        return probe_ollama(self._base_url)

    def load(self, spec: RunSpec) -> None:
        # The Ollama daemon loads the model lazily on first request; nothing to do.
        self.note = "served by the Ollama daemon (GGUF, OpenAI-compatible API)"

    def generate(self, spec: RunSpec) -> Iterator[str]:
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": spec.model_id,
            "messages": [{"role": "user", "content": spec.prompt}],
            "stream": True,
            "max_tokens": spec.max_new_tokens,
            "temperature": 0,
            "seed": spec.seed,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer ollama",
            },
            method="POST",
        )
        try:
            response = self._gatekeeper.execute(
                urllib.request.urlopen,  # noqa: S310 (localhost)
                request,
                timeout=120,
                retry_on=(TimeoutError, ConnectionError),
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:200]
            raise BackendUnavailable(
                self.name,
                f"HTTP {exc.code} from Ollama for {spec.model_id!r} "
                f"(is it pulled? `ollama pull {spec.model_id}`): {detail}",
            ) from exc
        except urllib.error.URLError as exc:
            raise BackendUnavailable(self.name, f"cannot reach Ollama: {exc.reason}") from exc

        with response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    yield delta
