from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import pdf_summarizer.config as config
from pdf_summarizer.config import Settings

FAKE_KEY = "AIza" + "x" * 35


def make_settings(**overrides) -> Settings:
    base: dict[str, object] = {
        "gemini_api_key": FAKE_KEY,
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_defaults_are_sane() -> None:
    settings = make_settings()
    assert settings.gemini_model
    assert settings.gemini_embedding_model
    assert 0.0 <= settings.temperature <= 2.0
    assert settings.chunk_overlap < settings.chunk_size
    assert settings.max_upload_bytes == settings.max_upload_mb * 1024 * 1024


def test_api_key_is_masked() -> None:
    masked = make_settings().masked_key
    assert masked.startswith("AIza")
    assert masked.endswith(FAKE_KEY[-4:])
    assert FAKE_KEY not in masked


def test_missing_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_placeholder_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(gemini_api_key="your_gemini_api_key_here")


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        make_settings(chunk_size=1000, chunk_overlap=1000)


def test_temperature_bounds() -> None:
    with pytest.raises(ValidationError):
        make_settings(temperature=3.0)


def test_env_diagnostics_reports_missing_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    diag = config.env_diagnostics()

    assert diag["expected .env"] == str(tmp_path / "missing.env")
    assert diag["file found"] == "NO"
    assert diag["GEMINI_API_KEY in shell env"] == "no"


def test_env_diagnostics_flags_key_in_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    assert config.env_diagnostics()["GEMINI_API_KEY in shell env"] == "yes"


def test_env_file_sources_cover_project_and_cwd() -> None:
    """A `.env` next to the checkout wins; `./.env` is the fallback."""
    sources = Settings.model_config["env_file"]
    assert sources == (".env", config.ENV_FILE)  # type: ignore[comparison-overlap]


def test_env_vars_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    monkeypatch.setenv("CHUNK_SIZE", "1234")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.gemini_model == "gemini-test-model"
    assert settings.chunk_size == 1234
