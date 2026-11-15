from __future__ import annotations

import pytest

from pdf_summarizer.loaders import load_pdf
from pdf_summarizer.utils import PdfError, ScannedPdfError


def test_reads_pages_with_one_based_numbers(sample_pdf_bytes: bytes) -> None:
    doc = load_pdf(sample_pdf_bytes, "sample.pdf")
    assert doc.page_count == 3
    assert [p.number for p in doc.pages] == [1, 2, 3]
    assert all(p.text for p in doc.pages)
    assert "Retrieval augmented generation" in doc.pages[0].text


def test_digest_is_stable_and_content_addressed(sample_pdf_bytes: bytes) -> None:
    first = load_pdf(sample_pdf_bytes, "a.pdf")
    second = load_pdf(sample_pdf_bytes, "b.pdf")
    assert first.digest == second.digest
    assert len(first.digest) == 64


def test_text_joins_non_empty_pages(sample_pdf_bytes: bytes) -> None:
    doc = load_pdf(sample_pdf_bytes)
    assert doc.text.count("Chunking strategy") == 1
    assert doc.char_count == sum(p.char_count for p in doc.pages)


def test_rejects_empty_bytes() -> None:
    with pytest.raises(PdfError, match="empty"):
        load_pdf(b"", "x.pdf")


def test_rejects_non_pdf_extension(sample_pdf_bytes: bytes) -> None:
    with pytest.raises(PdfError, match="not a PDF"):
        load_pdf(sample_pdf_bytes, "notes.txt")


def test_rejects_oversized_file(sample_pdf_bytes: bytes) -> None:
    with pytest.raises(PdfError, match="too large"):
        load_pdf(sample_pdf_bytes, "big.pdf", max_bytes=10)


def test_rejects_corrupt_bytes() -> None:
    with pytest.raises(PdfError):
        load_pdf(b"this is not a pdf at all", "broken.pdf")


def test_detects_scanned_pdf(blank_pdf_bytes: bytes) -> None:
    with pytest.raises(ScannedPdfError, match="scanned"):
        load_pdf(blank_pdf_bytes, "scan.pdf")
