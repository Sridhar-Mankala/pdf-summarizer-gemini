"""Summarisation engine: one call for short PDFs, map-reduce for long ones.

Design notes
------------
* Strategy is chosen from an estimated token budget, not page count, because a
  dense 5-page PDF can outgrow a light 40-page one.
* Every model call goes through :class:`_Caller`, so retries, error translation
  and call counting live in one place.
* ``model`` is injectable: tests use LangChain's fake chat model, production uses
  Gemini. Nothing here talks to the network unless a real model is passed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .chunking import Chunk
from .config import Settings
from .llm import build_chat_model
from .loaders import PdfDocument
from .prompts import (
    MAP_PROMPT,
    REDUCE_FINAL_PROMPT,
    REDUCE_PARTIAL_PROMPT,
    STUFF_PROMPT,
    SUMMARY_STRUCTURE,
    format_pages,
    length_guidance,
)
from .utils import AppError, EmptyDocumentError, estimate_tokens, friendly_model_error, get_logger

logger = get_logger("summarize")

Strategy = Literal["stuff", "map-reduce"]
ProgressCallback = Callable[[int, int, str], None]

DEFAULT_STUFF_TOKEN_LIMIT = 60_000
MAP_BATCH_TOKENS = 12_000
REDUCE_BATCH_TOKENS = 12_000
MAX_MERGE_PASSES = 8


class SummarizationError(AppError):
    """Summarisation failed in a way the user can act on."""


@dataclass(frozen=True, slots=True)
class SummaryResult:
    markdown: str
    strategy: Strategy
    pages: int
    chunks: int
    model_calls: int
    input_tokens: int
    elapsed_s: float

    @property
    def output_tokens(self) -> int:
        return estimate_tokens(self.markdown)


def choose_strategy(input_tokens: int, stuff_token_limit: int = DEFAULT_STUFF_TOKEN_LIMIT) -> Strategy:
    return "stuff" if input_tokens <= stuff_token_limit else "map-reduce"


def plan_calls(
    chunks: Sequence[Chunk], stuff_token_limit: int = DEFAULT_STUFF_TOKEN_LIMIT
) -> dict[str, int | str]:
    """Cheap pre-flight estimate the UI can show before spending quota."""
    tokens = sum(c.tokens for c in chunks)
    strategy = choose_strategy(tokens, stuff_token_limit)
    if strategy == "stuff":
        return {"strategy": strategy, "input_tokens": tokens, "estimated_calls": 1}
    batches = _group_by_tokens(chunks, MAP_BATCH_TOKENS, token_fn=lambda c: c.tokens)
    return {
        "strategy": strategy,
        "input_tokens": tokens,
        "estimated_calls": len(batches) + 1,  # maps + final reduce
    }


def _render_excerpt(chunks: Sequence[Chunk]) -> str:
    return "\n\n".join(f"[p. {chunk.page}]\n{chunk.text}" for chunk in chunks)


def _group_by_tokens[T](
    items: Sequence[T],
    budget: int,
    token_fn: Callable[[T], int],
) -> list[list[T]]:
    """Greedy packing so each model call stays under ``budget`` tokens."""
    groups: list[list[T]] = []
    current: list[T] = []
    used = 0
    for item in items:
        cost = token_fn(item)
        if current and used + cost > budget:
            groups.append(current)
            current, used = [], 0
        current.append(item)
        used += cost
    if current:
        groups.append(current)
    return groups


def _friendly_error(exc: Exception) -> str:
    return friendly_model_error(exc)


class _Caller:
    """Runs prompt | model | parser, counts calls and translates failures."""

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model
        self._parser = StrOutputParser()
        self.calls = 0

    def run(self, prompt: ChatPromptTemplate, **variables: object) -> str:
        self.calls += 1
        chain = prompt | self._model | self._parser
        try:
            raw = chain.invoke(variables)
        except Exception as exc:  # network, quota, validation - all become actionable
            logger.warning("model call %d failed: %s", self.calls, exc)
            raise SummarizationError(_friendly_error(exc)) from exc
        text = str(raw).strip()
        if not text:
            raise SummarizationError("Gemini returned an empty response. Try again or lower the temperature.")
        return text


def summarize(
    document: PdfDocument,
    chunks: Sequence[Chunk],
    settings: Settings | None = None,
    *,
    model: BaseChatModel | None = None,
    length: str = "standard",
    progress: ProgressCallback | None = None,
    stuff_token_limit: int = DEFAULT_STUFF_TOKEN_LIMIT,
) -> SummaryResult:
    """Summarise ``document`` using ``chunks`` (both produced by earlier phases)."""
    if not chunks:
        raise EmptyDocumentError("Nothing to summarise - the document has no usable text.")

    if model is None:
        if settings is None:
            raise ValueError("Pass either `model` or `settings` to summarize().")
        model = build_chat_model(settings)

    caller = _Caller(model)
    started = time.perf_counter()
    input_tokens = sum(chunk.tokens for chunk in chunks)
    strategy = choose_strategy(input_tokens, stuff_token_limit)

    logger.info(
        "summarising %s: %d pages, %d chunks, ~%d tokens, strategy=%s",
        document.filename,
        document.page_count,
        len(chunks),
        input_tokens,
        strategy,
    )

    if strategy == "stuff":
        markdown = caller.run(
            STUFF_PROMPT,
            page_count=document.page_count,
            structure=SUMMARY_STRUCTURE,
            length_guidance=length_guidance(length),
            document=_render_excerpt(chunks),
        )
        if progress:
            progress(1, 1, "summarised")
    else:
        markdown = _map_reduce(document, chunks, caller, length=length, progress=progress)

    elapsed = time.perf_counter() - started
    logger.info("summary done in %.1fs with %d model calls", elapsed, caller.calls)

    return SummaryResult(
        markdown=markdown,
        strategy=strategy,
        pages=document.page_count,
        chunks=len(chunks),
        model_calls=caller.calls,
        input_tokens=input_tokens,
        elapsed_s=round(elapsed, 2),
    )


def _map_reduce(
    document: PdfDocument,
    chunks: Sequence[Chunk],
    caller: _Caller,
    *,
    length: str,
    progress: ProgressCallback | None,
) -> str:
    batches = _group_by_tokens(chunks, MAP_BATCH_TOKENS, token_fn=lambda c: c.tokens)
    notes: list[str] = []

    for i, batch in enumerate(batches, start=1):
        pages = format_pages([chunk.page for chunk in batch])
        note = caller.run(MAP_PROMPT, pages=pages, text=_render_excerpt(batch))
        notes.append(note)
        if progress:
            progress(i, len(batches), "mapping")

    passes = 0
    while estimate_tokens("\n\n".join(notes)) > REDUCE_BATCH_TOKENS and passes < MAX_MERGE_PASSES:
        grouped = _group_by_tokens(notes, REDUCE_BATCH_TOKENS, estimate_tokens)
        if len(grouped) < 2:
            break
        merged: list[str] = []
        for i, group in enumerate(grouped, start=1):
            merged.append(
                group[0] if len(group) == 1 else caller.run(REDUCE_PARTIAL_PROMPT, notes="\n\n".join(group))
            )
            if progress:
                progress(i, len(grouped), "merging")
        if merged == notes:
            break
        notes = merged
        passes += 1

    return caller.run(
        REDUCE_FINAL_PROMPT,
        filename=document.filename,
        page_count=document.page_count,
        structure=SUMMARY_STRUCTURE,
        length_guidance=length_guidance(length),
        notes="\n\n".join(notes),
    )
