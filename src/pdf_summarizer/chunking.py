"""Split a parsed PDF into retrieval- and summary-ready chunks.

Chunking is per page so every chunk keeps an exact page number, which is what
makes citation ("p. 4") possible in the RAG phase. Pages longer than the chunk
size are split recursively on paragraph, line and sentence boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .loaders import PdfDocument
from .utils import estimate_tokens

SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


@dataclass(frozen=True, slots=True)
class Chunk:
    """A slice of one page, stably identified for caching and citations."""

    text: str
    page: int
    index: int  # position within the whole document
    page_index: int  # position within its page

    @property
    def id(self) -> str:
        return f"p{self.page}-c{self.page_index}"

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)

    @property
    def citation(self) -> str:
        return f"p. {self.page}"


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        keep_separator=True,
        strip_whitespace=True,
        length_function=len,
    )


def chunk_document(
    document: PdfDocument,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Split every non-empty page, preserving order and page numbers."""
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")

    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []

    for page in document.pages:
        if not page.text.strip():
            continue
        pieces = splitter.split_text(page.text)
        for page_index, piece in enumerate(pieces):
            text = piece.strip()
            if not text:
                continue
            chunks.append(Chunk(text=text, page=page.number, index=len(chunks), page_index=page_index))

    return chunks


def coverage_stats(document: PdfDocument, chunks: list[Chunk]) -> dict[str, int]:
    """Small summary the UI can display before spending tokens."""
    return {
        "pages": document.page_count,
        "chunks": len(chunks),
        "chars": sum(c.text.__len__() for c in chunks),
        "tokens": sum(c.tokens for c in chunks),
        "empty_pages": len(document.empty_pages),
    }
