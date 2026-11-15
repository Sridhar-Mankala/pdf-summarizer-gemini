from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from pdf_summarizer.chunking import chunk_document
from pdf_summarizer.loaders import load_pdf
from pdf_summarizer.summarize import (
    SummarizationError,
    SummaryResult,
    choose_strategy,
    plan_calls,
    summarize,
)
from pdf_summarizer.utils import EmptyDocumentError


class ExplodingModel(BaseChatModel):
    """Raises on every call so we can test error translation."""

    message: str = "429 RESOURCE_EXHAUSTED"

    @property
    def _llm_type(self) -> str:
        return "exploding"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError(self.message)


@pytest.fixture
def loaded(sample_pdf_bytes: bytes) -> tuple[Any, list[Any]]:
    doc = load_pdf(sample_pdf_bytes, "sample.pdf")
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
    return doc, chunks


def test_choose_strategy_boundaries() -> None:
    assert choose_strategy(0) == "stuff"
    assert choose_strategy(60_000) == "stuff"
    assert choose_strategy(60_001) == "map-reduce"
    assert choose_strategy(60_001, stuff_token_limit=100_000) == "stuff"


def test_plan_calls_for_small_document(loaded: tuple[Any, Sequence[Any]]) -> None:
    _doc, chunks = loaded
    plan = plan_calls(chunks)
    assert plan["strategy"] == "stuff"
    assert plan["estimated_calls"] == 1
    assert plan["input_tokens"] == sum(c.tokens for c in chunks)


def test_plan_calls_for_forced_map_reduce(loaded: tuple[Any, Sequence[Any]]) -> None:
    _, chunks = loaded
    plan = plan_calls(chunks, stuff_token_limit=0)
    assert plan["strategy"] == "map-reduce"
    assert int(plan["estimated_calls"]) >= 2


def test_stuff_summary_uses_one_call(loaded: tuple[Any, Sequence[Any]]) -> None:
    doc, chunks = loaded
    model = FakeListChatModel(responses=["# Final summary\n\nAll good."])

    result = summarize(doc, chunks, model=model)

    assert isinstance(result, SummaryResult)
    assert result.strategy == "stuff"
    assert result.model_calls == 1
    assert result.markdown == "# Final summary\n\nAll good."
    assert result.pages == doc.page_count
    assert result.chunks == len(chunks)
    assert result.input_tokens > 0
    assert result.elapsed_s >= 0
    assert result.output_tokens > 0


def test_map_reduce_runs_maps_then_reduce(loaded: tuple[Any, Sequence[Any]]) -> None:
    doc, chunks = loaded
    model = FakeListChatModel(responses=["- note about pages one to three [p. 1]", "# Reduced"])

    result = summarize(doc, chunks, model=model, stuff_token_limit=0)

    assert result.strategy == "map-reduce"
    assert result.model_calls == 2
    assert result.markdown == "# Reduced"


def test_progress_callback_reports_stages(loaded: tuple[Any, Sequence[Any]]) -> None:
    doc, chunks = loaded
    seen: list[tuple[int, int, str]] = []

    summarize(
        doc,
        chunks,
        model=FakeListChatModel(responses=["- a", "# done"]),
        stuff_token_limit=0,
        progress=lambda done, total, stage: seen.append((done, total, stage)),
    )

    assert seen
    stages = {stage for _, _, stage in seen}
    assert "mapping" in stages
    assert all(0 < done <= total for done, total, _ in seen)


def test_empty_chunks_rejected(loaded: tuple[Any, Sequence[Any]]) -> None:
    doc, _ = loaded
    with pytest.raises(EmptyDocumentError):
        summarize(doc, [], model=FakeListChatModel(responses=["x"]))


def test_missing_model_and_settings_rejected(loaded: tuple[Any, Sequence[Any]]) -> None:
    doc, chunks = loaded
    with pytest.raises(ValueError, match=r"model.*settings"):
        summarize(doc, chunks)


def test_rate_limit_error_is_translated(loaded: tuple[Any, Sequence[Any]]) -> None:
    doc, chunks = loaded
    with pytest.raises(SummarizationError, match=r"rate limit|quota"):
        summarize(doc, chunks, model=ExplodingModel())


def test_empty_model_response_rejected(loaded: tuple[Any, Sequence[Any]]) -> None:
    doc, chunks = loaded
    with pytest.raises(SummarizationError, match="empty"):
        summarize(doc, chunks, model=FakeListChatModel(responses=["   "]))


def test_missing_api_key_error_is_translated(loaded: tuple[Any, Sequence[Any]]) -> None:
    doc, chunks = loaded
    with pytest.raises(SummarizationError, match="API key"):
        summarize(doc, chunks, model=ExplodingModel(message="403 PERMISSION_DENIED: API key not valid"))


def test_ai_message_content_type_is_handled(loaded: tuple[Any, Sequence[Any]]) -> None:
    """FakeListChatModel returns AIMessage; make sure the parser path is exercised."""

    class Echo(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "echo"

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    doc, chunks = loaded
    result = summarize(doc, chunks, model=Echo())
    assert result.markdown == "ok"
