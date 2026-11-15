"""Retrieval-augmented Q&A over one parsed PDF, with page citations.

The index is built per document in memory and identified by a collection name
derived from the file digest, so two PDFs never share chunks. Models and
embeddings are injected, which keeps tests fully offline.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from .chunking import Chunk
from .prompts import RAG_PROMPT
from .utils import AppError, EmptyDocumentError, friendly_model_error, get_logger

logger = get_logger("rag")

DEFAULT_K = 4
PREVIEW_CHARS = 180
REFUSAL = "I couldn't find that in the document."
CITATION_RE = re.compile(r"\[p\.\s*(\d+)\]")


class RagError(AppError):
    """Retrieval or answering failed in a way the user can act on."""


@dataclass(frozen=True, slots=True)
class Citation:
    page: int
    chunk_id: str
    preview: str


@dataclass(frozen=True, slots=True)
class RagAnswer:
    answer: str
    citations: tuple[Citation, ...]
    retrieved: int

    @property
    def pages(self) -> tuple[int, ...]:
        return tuple(sorted({c.page for c in self.citations}))

    @property
    def is_refusal(self) -> bool:
        return self.answer.strip().rstrip(".").lower().startswith("i couldn't find that")


class PdfChat:
    """Ask questions about a single document; answers carry page citations."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        model: BaseChatModel,
        embeddings: Embeddings,
        k: int = DEFAULT_K,
        collection_name: str = "pdf-document",
    ) -> None:
        if not chunks:
            raise EmptyDocumentError("This document has no usable text to search.")
        self._k = max(1, k)
        self._chunk_count = len(chunks)
        self._chain = RAG_PROMPT | model | StrOutputParser()

        documents = [
            Document(
                page_content=chunk.text,
                metadata={"page": chunk.page, "chunk_id": chunk.id},
            )
            for chunk in chunks
        ]
        ids = [chunk.id for chunk in chunks]

        try:
            self._store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                client_settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._store.add_documents(documents, ids=ids)
        except Exception as exc:  # embedding quota / network / bad model
            logger.warning("index build failed: %s", exc)
            raise RagError(friendly_model_error(exc)) from exc

        logger.info("indexed %d chunks into %s", len(chunks), collection_name)

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    def ask(self, question: str, *, k: int | None = None) -> RagAnswer:
        question = question.strip()
        if not question:
            raise ValueError("Ask a question first.")

        try:
            retrieved = self._store.similarity_search(question, k=k or self._k)
        except Exception as exc:
            logger.warning("retrieval failed: %s", exc)
            raise RagError(friendly_model_error(exc)) from exc

        if not retrieved:
            return RagAnswer(REFUSAL, (), 0)

        context = "\n\n".join(f"[p. {doc.metadata['page']}] {doc.page_content}" for doc in retrieved)

        try:
            raw = self._chain.invoke({"context": context, "question": question})
        except Exception as exc:
            logger.warning("answer generation failed: %s", exc)
            raise RagError(friendly_model_error(exc)) from exc

        answer = str(raw).strip()
        if not answer:
            raise RagError("Gemini returned an empty answer. Try rephrasing the question.")

        citations = _citations(retrieved)
        if answer.rstrip(".").lower().startswith("i couldn't find that"):
            return RagAnswer(answer=answer, citations=(), retrieved=len(retrieved))

        cited_pages = {int(page) for page in CITATION_RE.findall(answer)}
        if cited_pages:
            relevant = tuple(c for c in citations if c.page in cited_pages)
            citations = relevant or citations

        return RagAnswer(answer=answer, citations=citations, retrieved=len(retrieved))


def _citations(documents: Sequence[Document]) -> tuple[Citation, ...]:
    seen: set[tuple[int, str]] = set()
    citations: list[Citation] = []
    for doc in documents:
        page = int(doc.metadata.get("page", 0))
        chunk_id = str(doc.metadata.get("chunk_id", ""))
        key = (page, chunk_id)
        if key in seen:
            continue
        seen.add(key)
        preview = " ".join(doc.page_content.split())[:PREVIEW_CHARS]
        citations.append(Citation(page=page, chunk_id=chunk_id, preview=preview))
    return tuple(citations)
