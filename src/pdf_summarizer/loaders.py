"""Turn uploaded PDF bytes into clean, page-addressable text.

Deliberately uses pypdf directly instead of a LangChain loader: we need exact
page numbers for citations and explicit control over failure modes.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .utils import PdfError, ScannedPdfError, clean_text, sha256_bytes

MIN_PAGE_CHARS = 10


@dataclass(frozen=True, slots=True)
class Page:
    """One PDF page with 1-based numbering, as humans count pages."""

    number: int
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True, slots=True)
class PdfDocument:
    filename: str
    pages: tuple[Page, ...]
    digest: str
    size_bytes: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(page.char_count for page in self.pages)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)

    @property
    def empty_pages(self) -> tuple[int, ...]:
        return tuple(page.number for page in self.pages if page.char_count < MIN_PAGE_CHARS)


def load_pdf(
    data: bytes,
    filename: str = "document.pdf",
    *,
    max_bytes: int | None = None,
) -> PdfDocument:
    """Parse PDF bytes into a :class:`PdfDocument`.

    Raises:
        PdfError: not a PDF, encrypted, corrupt, empty, or over the size limit.
        ScannedPdfError: valid PDF but no extractable text.
    """
    if not data:
        raise PdfError("The uploaded file is empty.")
    if not filename.lower().endswith(".pdf"):
        raise PdfError(f"'{filename}' is not a PDF file.")
    if max_bytes is not None and len(data) > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        raise PdfError(f"File is too large ({len(data) / (1024 * 1024):.1f} MB). Limit is {limit_mb:.0f} MB.")

    try:
        reader = PdfReader(io.BytesIO(data))
    except (PdfReadError, ValueError, OSError) as exc:
        raise PdfError(f"Could not read the PDF: {exc}") from exc

    if reader.is_encrypted:
        # Many PDFs are "encrypted" with an empty password - try that first.
        try:
            if not reader.decrypt(""):
                raise PdfError("This PDF is password protected. Remove the password and try again.")
        except Exception as exc:  # pypdf raises several types here
            raise PdfError("This PDF is password protected. Remove the password and try again.") from exc

    try:
        raw_pages = list(reader.pages)
    except (PdfReadError, ValueError, OSError) as exc:
        raise PdfError(f"Could not read the PDF pages: {exc}") from exc

    if not raw_pages:
        raise PdfError("The PDF has no pages.")

    pages: list[Page] = []
    for number, raw_page in enumerate(raw_pages, start=1):
        try:
            extracted = raw_page.extract_text() or ""
        except Exception as exc:  # a single bad page should not kill the run
            raise PdfError(f"Page {number} could not be parsed: {exc}") from exc
        pages.append(Page(number=number, text=clean_text(extracted)))

    total_chars = sum(page.char_count for page in pages)
    if total_chars < MIN_PAGE_CHARS:
        raise ScannedPdfError(
            "No extractable text found. This looks like a scanned or image-only PDF - OCR is not supported yet."
        )

    meta = {}
    try:
        for key, value in (reader.metadata or {}).items():
            if value is not None:
                meta[str(key).lstrip("/")] = str(value)
    except Exception:  # metadata is best effort
        meta = {}

    return PdfDocument(
        filename=filename,
        pages=tuple(pages),
        digest=sha256_bytes(data),
        size_bytes=len(data),
        metadata=meta,
    )
