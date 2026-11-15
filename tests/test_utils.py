from __future__ import annotations

from pdf_summarizer.utils import clean_text, estimate_tokens, human_bytes, sha256_bytes


def test_clean_text_collapses_runs() -> None:
    assert clean_text("a   b\t\tc") == "a b c"


def test_clean_text_normalises_newlines() -> None:
    assert clean_text("line1\r\nline2\r\n\r\n\r\n\r\nline3") == "line1\nline2\n\nline3"


def test_clean_text_joins_hyphenated_line_breaks() -> None:
    assert clean_text("retrie-\nval") == "retrieval"


def test_clean_text_strips_line_edges() -> None:
    assert clean_text("  padded  \n  next  ") == "padded\nnext"


def test_estimate_tokens_is_positive_and_monotonic() -> None:
    assert estimate_tokens("") == 0
    short = estimate_tokens("hello world")
    longer = estimate_tokens("hello world " * 10)
    assert 0 < short < longer


def test_sha256_is_hex_and_stable() -> None:
    digest = sha256_bytes(b"abc")
    assert digest == sha256_bytes(b"abc")
    assert len(digest) == 64


def test_human_bytes() -> None:
    assert human_bytes(512).endswith("B")
    assert human_bytes(2048) == "2.0 KB"
    assert human_bytes(5 * 1024 * 1024) == "5.0 MB"
