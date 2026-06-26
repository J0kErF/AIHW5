"""Pull models from the Hugging Face Hub (docs/SPECIFICATION.md §3.1).

Downloads weights (SafeTensors preferred) into the localforge cache, records the
result in the registry, and turns auth/availability failures into typed errors.
Re-pulling an already-present model is a no-op.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from localforge.config.settings import Settings
from localforge.core.errors import AuthError, ConfigError
from localforge.core.gatekeeper import ApiGatekeeper, RateLimitConfig
from localforge.core.logging import get_logger, register_secret
from localforge.models.formats import detect_format, weight_bytes
from localforge.models.registry import ModelInfo, ModelRegistry

_log = get_logger(__name__)


def pull_model(model_id: str, settings: Settings, *, force: bool = False) -> ModelInfo:
    """Ensure ``model_id`` is available locally and return its :class:`ModelInfo`.

    Raises :class:`AuthError` for gated/unauthorized models and
    :class:`ConfigError` for a missing repo.
    """
    registry = ModelRegistry(settings.cache_dir)
    cached = registry.get(model_id)
    if cached is not None and not force and Path(cached.path).exists():
        _log.info("model %s already present (%.0f MB)", model_id, cached.size_mb)
        return cached

    token: str | None = None
    if settings.hf_token is not None:
        token = settings.hf_token.get_secret_value()
        register_secret(token)

    hf_cache = settings.cache_dir / "hf"
    # Route the download through the gatekeeper: retry transient network errors.
    gatekeeper = ApiGatekeeper(RateLimitConfig(max_retries=2, backoff_s=1.0))
    try:
        local_dir = Path(
            gatekeeper.execute(
                snapshot_download,
                repo_id=model_id,
                token=token,
                cache_dir=str(hf_cache),
                # huggingface_hub uses httpx; retry transient transport failures
                # (connect/read errors), not HTTP status errors (handled below).
                retry_on=(httpx.TransportError, TimeoutError),
            )
        )
    except GatedRepoError as exc:
        raise AuthError(
            f"{model_id} is gated — accept its license on the Hub and set HF_TOKEN"
        ) from exc
    except RepositoryNotFoundError as exc:
        raise ConfigError(f"model not found on the Hub: {model_id}") from exc
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise AuthError(f"not authorized for {model_id}; set a valid HF_TOKEN in .env") from exc
        raise

    files = [str(p.relative_to(local_dir)) for p in local_dir.rglob("*") if p.is_file()]
    info = ModelInfo(
        model_id=model_id,
        format=detect_format(files),
        size_bytes=weight_bytes(local_dir),
        path=str(local_dir),
    )
    registry.record(info)
    _log.info("pulled %s [%s, %.0f MB]", model_id, info.format, info.size_mb)
    return info
