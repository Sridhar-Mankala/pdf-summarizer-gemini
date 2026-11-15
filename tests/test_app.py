from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_app_renders_without_exception() -> None:
    """The app must survive a cold run, even with a broken/missing config."""
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert at.title


def test_app_has_uploader() -> None:
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    uploaders = at.get("file_uploader")
    assert len(uploaders) == 1
