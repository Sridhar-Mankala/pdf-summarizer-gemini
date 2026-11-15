"""Single source of truth for runtime configuration.

Values come from the environment or the project `.env` file. Field names map to
upper-case env vars automatically (``chunk_size`` <- ``CHUNK_SIZE``), and a
missing API key fails fast instead of surfacing as a confusing 403 later.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    """Validated application settings loaded from env / .env."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- credentials -----------------------------------------------------
    gemini_api_key: SecretStr

    # --- models ----------------------------------------------------------
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # --- ingestion -------------------------------------------------------
    max_upload_mb: int = Field(default=25, ge=1, le=200)
    chunk_size: int = Field(default=4000, ge=500, le=200_000)
    chunk_overlap: int = Field(default=200, ge=0)

    # --- runtime ---------------------------------------------------------
    log_level: LogLevel = "INFO"

    @field_validator("gemini_api_key")
    @classmethod
    def _reject_placeholder(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value().strip()
        if not raw:
            raise ValueError("GEMINI_API_KEY is empty - get a key at https://aistudio.google.com/apikey")
        if raw.lower().startswith(("your_", "changeme", "<")):
            raise ValueError("GEMINI_API_KEY still holds the placeholder from .env.example")
        return SecretStr(raw)

    @model_validator(mode="after")
    def _check_chunking(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than CHUNK_SIZE ({self.chunk_size})"
            )
        return self

    # --- helpers ---------------------------------------------------------
    @property
    def api_key(self) -> str:
        """Plain key for SDK clients. Never log this."""
        return self.gemini_api_key.get_secret_value()

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def masked_key(self) -> str:
        raw = self.api_key
        return f"{raw[:4]}...{raw[-4:]}" if len(raw) > 8 else "*" * len(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance - import-safe and cheap to call repeatedly."""
    return Settings()  # type: ignore[call-arg]  # values come from env/.env
