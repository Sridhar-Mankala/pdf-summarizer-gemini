from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from pdf_summarizer.chunking import chunk_document
from pdf_summarizer.loaders import load_pdf
from pdf_summarizer.rag import Citation, PdfChat, RagError, _citations
from pdf_summarizer.utils import EmptyDocumentError


class ExplodingModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "exploding"

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
        raise RuntimeError("429 RESOURCE_EXHAUSTED")


def make_embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=16)


@pytest.fixture
def loaded(sample_pdf_bytes: bytes) -> tuple[Any, list[Any]]:
    doc = load_pdf(sample_pdf_bytes, "sample.pdf")
    return doc, chunk_document(doc, chunk_size=200, chunk_overlap=20)


def test_ask_returns_answer_with_citations(loaded: tuple[Any, Sequence[Any]]) -> None:
    _, chunks = loaded
    chat = PdfChat(
        chunks,
        model=FakeListChatModel(responses=["Chunking keeps citations exact [p. 2]."]),
        embeddings=make_embeddings(),
        k=3,
        collection_name="t_ask",
    )

    answer = chat.ask("Why does chunking matter?")

    assert answer.retrieved == 3
    assert "p. 2" in answer.answer
    assert answer.citations
    assert answer.pages == (2,)
    assert all(c.preview for c in answer.citations)
    assert not answer.is_refusal


def test_k_controls_retrieval_size(loaded: tuple[Any, Sequence[Any]]) -> None:
    _, chunks = loaded
    chat = PdfChat(
        chunks,
        model=FakeListChatModel(responses=["one source [p. 1]"]),
        embeddings=make_embeddings(),
        collection_name="t_k",
    )
    assert chat.ask("anything", k=1).retrieved == 1


def test_refusal_is_detected(loaded: tuple[Any, Sequence[Any]]) -> None:
    _, chunks = loaded
    chat = PdfChat(
        chunks,
        model=FakeListChatModel(responses=["I couldn't find that in the document."]),
        embeddings=make_embeddings(),
        collection_name="t_refusal",
    )
    assert chat.ask("What is the capital of France?").is_refusal


def test_refusal_has_no_citations(loaded: tuple[Any, Sequence[Any]]) -> None:
    _, chunks = loaded
    chat = PdfChat(
        chunks,
        model=FakeListChatModel(responses=["I couldn't find that in the document."]),
        embeddings=make_embeddings(),
        collection_name="t_refusal_cites",
    )
    answer = chat.ask("Unrelated question about penguins?")
    assert answer.citations == ()
    assert answer.pages == ()


def test_empty_question_rejected(loaded: tuple[Any, Sequence[Any]]) -> None:
    _, chunks = loaded
    chat = PdfChat(
        chunks,
        model=FakeListChatModel(responses=["x"]),
        embeddings=make_embeddings(),
        collection_name="t_empty_q",
    )
    with pytest.raises(ValueError, match="question"):
        chat.ask("   ")


def test_empty_chunks_rejected() -> None:
    with pytest.raises(EmptyDocumentError):
        PdfChat([], model=FakeListChatModel(responses=["x"]), embeddings=make_embeddings())


def test_model_failure_becomes_rag_error(loaded: tuple[Any, Sequence[Any]]) -> None:
    _, chunks = loaded
    chat = PdfChat(chunks, model=ExplodingModel(), embeddings=make_embeddings(), collection_name="t_err")
    with pytest.raises(RagError, match=r"rate limit|quota"):
        chat.ask("hello?")


def test_citations_deduplicate() -> None:
    docs = [
        Document(page_content="alpha", metadata={"page": 1, "chunk_id": "p1-c0"}),
        Document(page_content="beta", metadata={"page": 2, "chunk_id": "p2-c0"}),
        Document(page_content="alpha again", metadata={"page": 1, "chunk_id": "p1-c0"}),
    ]
    citations = _citations(docs)
    assert len(citations) == 2
    assert {c.page for c in citations} == {1, 2}
    assert isinstance(citations[0], Citation)


def test_preview_is_single_line_and_bounded() -> None:
    docs = [Document(page_content="word " * 100, metadata={"page": 5, "chunk_id": "p5-c0"})]
    preview = _citations(docs)[0].preview
    assert "\n" not in preview
    assert len(preview) <= 180
