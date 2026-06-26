"""Unit tests for HF model acquisition with snapshot_download mocked."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

import localforge.models.acquire as acq
from localforge.config.settings import load_settings
from localforge.core.errors import AuthError, ConfigError
from localforge.models.formats import ModelFormat


def _settings(tmp_path: Path):
    return load_settings(cache_dir=tmp_path, hf_token="tok_secret_value")


def _fake_model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "snap"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"w" * 2048)
    (d / "config.json").write_text("{}", encoding="utf-8")
    return d


def test_pull_success_records_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snap = _fake_model_dir(tmp_path)
    calls = {"n": 0}

    def fake_download(**kwargs: object) -> str:
        calls["n"] += 1
        return str(snap)

    monkeypatch.setattr(acq, "snapshot_download", fake_download)
    info = acq.pull_model("org/model", _settings(tmp_path))
    assert info.format is ModelFormat.SAFETENSORS
    assert info.size_bytes == 2048
    assert calls["n"] == 1

    # Re-pull is a no-op: cached in the registry, snapshot_download not called again.
    info2 = acq.pull_model("org/model", _settings(tmp_path))
    assert info2.model_id == "org/model"
    assert calls["n"] == 1


def test_pull_gated_raises_auth_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_gated(**kwargs: object):
        raise acq.GatedRepoError.__new__(acq.GatedRepoError)

    monkeypatch.setattr(acq, "snapshot_download", raise_gated)
    with pytest.raises(AuthError):
        acq.pull_model("org/gated", _settings(tmp_path))


def test_pull_missing_repo_raises_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_missing(**kwargs: object):
        raise acq.RepositoryNotFoundError.__new__(acq.RepositoryNotFoundError)

    monkeypatch.setattr(acq, "snapshot_download", raise_missing)
    with pytest.raises(ConfigError):
        acq.pull_model("org/missing", _settings(tmp_path))


def test_pull_unauthorized_raises_auth_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_401(**kwargs: object):
        exc = acq.HfHubHTTPError.__new__(acq.HfHubHTTPError)
        exc.response = types.SimpleNamespace(status_code=401)  # type: ignore[attr-defined]
        raise exc

    monkeypatch.setattr(acq, "snapshot_download", raise_401)
    with pytest.raises(AuthError):
        acq.pull_model("org/private", _settings(tmp_path))
