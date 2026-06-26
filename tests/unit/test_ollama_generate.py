"""Unit tests for the Ollama backend generate() path (urllib mocked)."""

from __future__ import annotations

import io
import urllib.error
import urllib.request

import pytest

from localforge.backends.ollama_backend import OllamaBackend
from localforge.core.errors import BackendUnavailable
from localforge.core.types import Backend, RunSpec


class _FakeResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def __iter__(self):
        return iter(self._lines)


def _spec() -> RunSpec:
    return RunSpec(model_id="qwen2.5:0.5b", backend=Backend.OLLAMA, prompt="hi", max_new_tokens=8)


def test_generate_parses_sse_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        b'data: {"choices":[{"delta":{"content":"Virtual"}}]}\n',
        b"\n",
        b'data: {"choices":[{"delta":{"content":" memory"}}]}\n',
        b"data: [DONE]\n",
    ]
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResponse(lines))
    out = "".join(OllamaBackend().generate(_spec()))
    assert out == "Virtual memory"


def test_generate_http_error_becomes_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_http(*a: object, **k: object):
        raise urllib.error.HTTPError(
            "http://x",
            404,
            "Not Found",
            {},
            io.BytesIO(b'{"error":"no model"}'),  # type: ignore[arg-type]
        )

    monkeypatch.setattr(urllib.request, "urlopen", raise_http)
    with pytest.raises(BackendUnavailable):
        list(OllamaBackend().generate(_spec()))


def test_generate_url_error_becomes_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_url(*a: object, **k: object):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", raise_url)
    with pytest.raises(BackendUnavailable):
        list(OllamaBackend().generate(_spec()))
