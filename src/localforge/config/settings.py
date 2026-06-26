"""Typed application settings.

Precedence (high -> low): explicit init args (CLI flags) > environment variables
> ``.env`` file > ``config/settings.toml`` defaults. See docs/IMPLEMENTATION.md §8.
The Hugging Face token is a ``SecretStr`` so it never appears in logs/reprs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SETTINGS_TOML = _PROJECT_ROOT / "config" / "settings.toml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        toml_file=_SETTINGS_TOML,
    )

    hf_token: SecretStr | None = Field(default=None, alias="HF_TOKEN")
    ollama_base_url: str = Field(default="http://localhost:11434/v1", alias="OLLAMA_BASE_URL")
    cache_dir: Path = Field(default=Path.home() / ".cache" / "localforge", alias="LOCALFORGE_CACHE")
    airllm_ram_ceiling_mb: int = 4096
    default_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    showcase_model: str = "Qwen/Qwen2.5-3B-Instruct"
    seed: int = 0

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # First source wins -> init > env > .env > settings.toml.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
        )


def load_settings(**overrides: Any) -> Settings:
    """Load settings, applying optional explicit overrides (highest precedence)."""
    return Settings(**overrides)
