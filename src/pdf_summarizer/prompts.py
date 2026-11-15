"""Prompt templates for summarisation.

All prompts are grounded: they forbid outside knowledge and ask the model to keep
page markers so later steps (and the UI) can point back to the source.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate

SummaryLength = Literal["brief", "standard", "detailed"]

LENGTH_GUIDANCE: dict[str, str] = {
    "brief": "about 150 words total, ruthlessly prioritised",
    "standard": "300-400 words total",
    "detailed": "600-800 words total, with section notes fully expanded",
}

SUMMARY_SYSTEM = (
    "You are a precise document analyst. Summarise ONLY what the source says. "
    "Never add outside knowledge, never invent facts, figures or names. "
    "Preserve numbers, dates and proper nouns exactly as written. "
    "Use markdown. Page markers such as [p. 3] come from the source; keep them "
    "when you reference something specific."
)

SUMMARY_STRUCTURE = """## TL;DR
Two or three sentences capturing the document's purpose and main claim.

## Key points
- 5 to 8 bullets, most important first.

## Section notes
Short paragraphs following the document's own flow.

## Action items & numbers
Anything actionable, plus the key figures, dates and names. Write "None stated" if absent.

## Open questions
Gaps, caveats or things the document does not resolve. Write "None" if the document is complete."""

STUFF_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SUMMARY_SYSTEM),
        (
            "human",
            """Summarise the document below ({page_count} pages, excerpt markers kept).

Use exactly this structure:

{structure}

Target length: {length_guidance}

<document>
{document}
</document>""",
        ),
    ]
)

MAP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract faithful notes from one excerpt of a larger document. "
            "Bullets only, no preamble, no conclusions that the excerpt does not support.",
        ),
        (
            "human",
            """Source pages: {pages}

Write 4-8 bullets capturing the substance of this excerpt. Keep any numbers, names
and dates. Tag each bullet with the page it came from, e.g. [p. 4]. If the excerpt is
mostly boilerplate (references, headers), say "Low information: <one line>".

<excerpt>
{text}
</excerpt>""",
        ),
    ]
)

REDUCE_PARTIAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You merge sets of notes into tighter notes. Keep every distinct fact, drop "
            "duplication and pagekeeping. Bullets only. Never invent content.",
        ),
        ("human", "Merge these notes, preserving page tags:\n\n{notes}"),
    ]
)

REDUCE_FINAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SUMMARY_SYSTEM),
        (
            "human",
            """Below are notes extracted from every page of "{filename}" ({page_count} pages).
Turn them into the final summary. Do not invent anything that is not in the notes.

Use exactly this structure:

{structure}

Target length: {length_guidance}

<notes>
{notes}
</notes>""",
        ),
    ]
)


def length_guidance(length: str) -> str:
    return LENGTH_GUIDANCE.get(length, LENGTH_GUIDANCE["standard"])


RAG_SYSTEM = (
    "You answer questions about ONE document using only the excerpts provided. "
    "Every factual sentence must carry an inline page citation like [p. 3]. "
    "If the excerpts do not contain the answer, reply exactly: "
    '"I couldn\'t find that in the document." '
    "Never use outside knowledge and never guess."
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM),
        (
            "human",
            """Excerpts from the document:
{context}

Question: {question}

Answer concisely, with inline [p. N] citations after each claim.""",
        ),
    ]
)


def format_pages(pages: list[int]) -> str:
    if not pages:
        return "unknown"
    ordered = sorted(set(pages))
    if len(ordered) == 1:
        return f"p. {ordered[0]}"
    return f"pp. {ordered[0]}-{ordered[-1]}"
