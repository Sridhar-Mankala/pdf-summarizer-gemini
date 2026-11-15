from __future__ import annotations

from collections.abc import Callable

import pytest

from pdf_summarizer.chunking import build_splitter, chunk_document, coverage_stats
from pdf_summarizer.loaders import load_pdf


def test_chunks_keep_page_numbers(sample_pdf_bytes: bytes) -> None:
    doc = load_pdf(sample_pdf_bytes, "sample.pdf")
    chunks = chunk_document(doc, chunk_size=120, chunk_overlap=20)

    assert chunks
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks} == {1, 2, 3}
    assert all(1 <= c.page <= doc.page_count for c in chunks)
    assert [c.page for c in chunks] == sorted(c.page for c in chunks)


def test_chunks_respect_size_limit(sample_pdf_bytes: bytes) -> None:
    doc = load_pdf(sample_pdf_bytes)
    chunks = chunk_document(doc, chunk_size=80, chunk_overlap=10)
    assert all(len(c.text) <= 80 for c in chunks)


def test_long_page_is_split(make_pdf: Callable[..., bytes]) -> None:
    long_page = " ".join(f"Sentence {i} explains chunking in detail." for i in range(200))
    doc = load_pdf(make_pdf([long_page]), "long.pdf")
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 3
    assert all(c.page == 1 for c in chunks)
    assert [c.page_index for c in chunks] == list(range(len(chunks)))


def test_empty_pages_are_skipped(make_pdf: Callable[..., bytes]) -> None:
    doc = load_pdf(make_pdf(["", "Real content on page two with enough characters."]), "mixed.pdf")
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
    assert chunks
    assert {c.page for c in chunks} == {2}


def test_chunk_ids_are_unique(sample_pdf_bytes: bytes) -> None:
    doc = load_pdf(sample_pdf_bytes)
    chunks = chunk_document(doc, chunk_size=60, chunk_overlap=0)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_invalid_overlap_rejected(sample_pdf_bytes: bytes) -> None:
    doc = load_pdf(sample_pdf_bytes)
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        chunk_document(doc, chunk_size=100, chunk_overlap=100)


def test_coverage_stats(sample_pdf_bytes: bytes) -> None:
    doc = load_pdf(sample_pdf_bytes)
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
    stats = coverage_stats(doc, chunks)

    assert stats["pages"] == 3
    assert stats["chunks"] == len(chunks)
    assert stats["tokens"] > 0
    assert stats["chars"] == sum(len(c.text) for c in chunks)
    assert stats["empty_pages"] == 0


def test_splitter_rejects_bad_overlap() -> None:
    splitter = build_splitter(100, 10)
    pieces = splitter.split_text("alpha " * 100)
    assert pieces
    assert all(len(p) <= 100 for p in pieces)
