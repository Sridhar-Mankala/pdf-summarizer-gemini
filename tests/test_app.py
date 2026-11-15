from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"
FAKE_KEY = "AIza" + "x" * 35


@pytest.fixture
def fake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI has no .env, so give the app a valid-looking config via the environment."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    monkeypatch.delenv("GEMINI_EMBEDDING_MODEL", raising=False)


def test_app_renders_without_exception(fake_env: None) -> None:
    """The app must survive a cold run without raising."""
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert at.title


def test_app_has_uploader(fake_env: None) -> None:
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    uploaders = at.get("file_uploader")
    assert len(uploaders) == 1
