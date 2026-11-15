"""Streamlit front-end: upload a PDF, summarise it, download the result.

State handling is digest-based: every derived object (parsed pages, chunks,
summary) is keyed to the SHA-256 of the uploaded bytes, so Streamlit reruns and
widget changes never mix results from two different documents.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pdf_summarizer.chunking import Chunk, chunk_document, coverage_stats  # noqa: E402
from pdf_summarizer.config import Settings, get_settings  # noqa: E402
from pdf_summarizer.llm import build_chat_model, build_embeddings  # noqa: E402
from pdf_summarizer.loaders import PdfDocument, load_pdf  # noqa: E402
from pdf_summarizer.rag import PdfChat  # noqa: E402
from pdf_summarizer.summarize import SummaryResult, plan_calls, summarize  # noqa: E402
from pdf_summarizer.utils import AppError, configure_logging, human_bytes, sha256_bytes  # noqa: E402

LENGTHS = {"Brief": "brief", "Standard": "standard", "Detailed": "detailed"}


@st.cache_resource(show_spinner=False)
def cached_model(model_name: str, temperature: float):
    """One client per (model, temperature) pair, shared across sessions."""
    settings = get_settings()
    tuned = settings.model_copy(update={"gemini_model": model_name, "temperature": temperature})
    return build_chat_model(tuned)


def process_pdf(
    data: bytes,
    filename: str,
    settings: Settings,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[PdfDocument, list[Chunk]]:
    document = load_pdf(data, filename, max_bytes=settings.max_upload_bytes)
    return document, chunk_document(document, chunk_size, chunk_overlap)


def get_processed(
    data: bytes,
    filename: str,
    settings: Settings,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[PdfDocument, list[Chunk]]:
    digest = sha256_bytes(data)
    key = f"{digest}:{chunk_size}:{chunk_overlap}"
    cached = st.session_state.get("processed")
    if cached and cached["key"] == key:
        return cached["document"], cached["chunks"]

    document, chunks = process_pdf(data, filename, settings, chunk_size, chunk_overlap)
    st.session_state["processed"] = {"key": key, "document": document, "chunks": chunks}
    return document, chunks


def render_sidebar(settings: Settings) -> tuple[str, str, float, int, int]:
    with st.sidebar:
        st.header("Settings")
        model_name = st.text_input(
            "Gemini model", value=settings.gemini_model, help="Must be available to your API key."
        )
        length_label = st.radio("Summary length", list(LENGTHS), index=1, horizontal=False)
        temperature = st.slider("Temperature", 0.0, 1.0, float(settings.temperature), 0.05)

        with st.expander("Chunking (advanced)"):
            chunk_size = st.number_input(
                "Chunk size (chars)", min_value=500, max_value=20_000, value=settings.chunk_size, step=250
            )
            chunk_overlap = st.number_input(
                "Chunk overlap (chars)", min_value=0, max_value=2_000, value=settings.chunk_overlap, step=50
            )
            if chunk_overlap >= chunk_size:
                st.warning("Overlap must be smaller than chunk size - using defaults.")
                chunk_size, chunk_overlap = settings.chunk_size, settings.chunk_overlap

        st.divider()
        st.caption(f"Embedding model: `{settings.gemini_embedding_model}`")
        st.caption(f"API key: `{settings.masked_key}` · max upload {settings.max_upload_mb} MB")

    return model_name, LENGTHS[length_label], temperature, int(chunk_size), int(chunk_overlap)


def render_overview(document: PdfDocument, chunks: list[Chunk], settings: Settings) -> None:
    stats = coverage_stats(document, chunks)
    plan = plan_calls(chunks)

    cols = st.columns(4)
    cols[0].metric("Pages", stats["pages"])
    cols[1].metric("Chunks", stats["chunks"])
    cols[2].metric("Input tokens (est.)", f"~{stats['tokens']:,}")
    cols[3].metric("Gemini calls (est.)", plan["estimated_calls"])

    st.caption(
        f"{document.filename} · {human_bytes(document.size_bytes)} · strategy **{plan['strategy']}**"
        + (f" · {stats['empty_pages']} page(s) had no text" if stats["empty_pages"] else "")
    )
    if plan["strategy"] == "map-reduce":
        st.info(
            "Long document: it will be summarised in parts, then merged. This costs several Gemini calls."
        )


def render_summary(result: SummaryResult, document: PdfDocument) -> None:
    st.markdown(result.markdown)
    st.divider()
    meta = st.columns(4)
    meta[0].metric("Strategy", result.strategy)
    meta[1].metric("Model calls", result.model_calls)
    meta[2].metric("Time", f"{result.elapsed_s:.1f}s")
    meta[3].metric("Output tokens (est.)", f"~{result.output_tokens:,}")

    st.download_button(
        "Download summary (.md)",
        data=result.markdown,
        file_name=f"{Path(document.filename).stem}-summary.md",
        mime="text/markdown",
        use_container_width=False,
    )


def render_ask_tab(
    chunks: list[Chunk],
    settings: Settings,
    model_name: str,
    temperature: float,
    digest: str,
) -> None:
    """Chat over the document; the index is built once per document digest."""
    state = st.session_state.get("chat")
    history = state["history"] if state and state["key"] == digest else []

    for question, answer in history:
        st.chat_message("user").write(question)
        with st.chat_message("assistant"):
            st.markdown(answer.answer)
            if answer.citations:
                pages = ", ".join(f"p. {page}" for page in answer.pages)
                with st.expander(f"Sources ({pages})"):
                    for citation in answer.citations:
                        st.markdown(f"**p. {citation.page}** - _{citation.preview}_")

    if history and st.button("Clear conversation", key=f"clear-{digest[:8]}"):
        st.session_state.pop("chat", None)
        st.rerun()

    question = st.text_input(
        "Ask a question about this PDF",
        key=f"question-{digest[:8]}",
        placeholder="What does the document say about ...?",
    )
    if not st.button("Ask", type="primary", key=f"ask-{digest[:8]}") or not question.strip():
        if not history:
            st.caption("Answers use only this PDF and cite the page numbers they come from.")
        return

    current = st.session_state.get("chat")
    if not current or current["key"] != digest:
        with st.spinner("Indexing pages with Gemini embeddings..."):
            try:
                chat = PdfChat(
                    chunks,
                    model=cached_model(model_name, temperature),
                    embeddings=build_embeddings(settings),
                    collection_name=f"pdf-{digest[:12]}",
                )
            except AppError as exc:
                st.error(str(exc))
                return
        current = {"key": digest, "chat": chat, "history": []}
        st.session_state["chat"] = current

    try:
        with st.spinner("Retrieving and answering..."):
            answer = current["chat"].ask(question)
    except AppError as exc:
        st.error(str(exc))
        return

    current["history"].append((question, answer))
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="PDF Summarizer", page_icon="📄", layout="wide")
    st.title("📄 PDF Summarizer")
    st.caption("LangChain · Google Gemini · grounded, structured summaries")

    try:
        settings = get_settings()
        configure_logging(settings.log_level)
    except Exception as exc:
        st.error("Configuration problem - fix `.env` and reload.")
        st.code(str(exc), language="text")
        st.stop()

    model_name, length, temperature, chunk_size, chunk_overlap = render_sidebar(settings)

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is None:
        st.info(
            "Drop a PDF above. You'll get a TL;DR, key points, section notes, numbers and open questions."
        )
        st.stop()

    data = uploaded.getvalue()
    try:
        with st.spinner("Reading the PDF..."):
            document, chunks = get_processed(
                data, uploaded.name, settings, int(chunk_size), int(chunk_overlap)
            )
    except AppError as exc:
        st.error(str(exc))
        st.stop()

    if not chunks:
        st.error("This PDF produced no usable text chunks.")
        st.stop()

    render_overview(document, chunks, settings)

    digest = sha256_bytes(data)
    stored = st.session_state.get("summary")
    is_current = bool(stored and stored["key"] == digest)

    if st.button("Summarise", type="primary", use_container_width=True):
        progress = st.progress(0.0, text="Starting...")

        def on_progress(done: int, total: int, stage: str) -> None:
            progress.progress(min(done / max(total, 1), 1.0), text=f"{stage.capitalize()} {done}/{total}")

        try:
            result = summarize(
                document,
                chunks,
                model=cached_model(model_name, temperature),
                length=length,
                progress=on_progress,
            )
        except AppError as exc:
            progress.empty()
            st.error(str(exc))
            st.stop()
        except Exception as exc:  # unexpected - still show something readable
            progress.empty()
            st.error(f"Unexpected error: {exc}")
            st.stop()

        progress.empty()
        st.session_state["summary"] = {"key": digest, "result": result}
        is_current = True
        st.success(f"Done in {result.elapsed_s:.1f}s across {result.model_calls} Gemini call(s).")

    summary_tab, ask_tab, pages_tab = st.tabs(["Summary", "Ask", "Pages"])
    with summary_tab:
        if is_current:
            render_summary(st.session_state["summary"]["result"], document)
        else:
            st.caption("Press **Summarise** to generate a summary. Settings changes apply on the next run.")

    with ask_tab:
        render_ask_tab(chunks, settings, model_name, temperature, digest)

    with pages_tab:
        labels = [f"Page {page.number} ({page.char_count} chars)" for page in document.pages]
        choice = st.selectbox("Page", labels)
        page = document.pages[labels.index(choice)]
        st.text_area("Extracted text", page.text or "(no text on this page)", height=320, disabled=True)


if __name__ == "__main__":
    main()
