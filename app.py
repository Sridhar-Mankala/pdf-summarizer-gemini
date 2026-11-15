from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pdf_summarizer.config import get_settings  # noqa: E402

PHASES = [
    ("0", "Environment, API key and GitHub auth", "done"),
    ("1", "Skeleton, dependencies, config layer", "done"),
    ("2", "PDF ingestion and chunking", "next"),
    ("3", "Summarization engine", "planned"),
    ("4", "Streamlit UI", "planned"),
    ("5", "RAG Q&A with citations", "planned"),
    ("6", "Tests, lint, CI", "planned"),
]


def main() -> None:
    st.set_page_config(page_title="PDF Summarizer", page_icon="📄", layout="wide")
    st.title("📄 PDF Summarizer + RAG")
    st.caption("Phase 1 skeleton - upload, summaries and chat arrive in later phases.")

    try:
        settings = get_settings()
    except Exception as exc:  # configuration problems should be readable, not a stack trace
        st.error("Configuration error - fix `.env` and reload.")
        st.code(str(exc), language="text")
        st.stop()

    st.subheader("Runtime configuration")
    left, right = st.columns(2)
    with left:
        st.metric("Chat model", settings.gemini_model)
        st.metric("Embedding model", settings.gemini_embedding_model)
        st.metric("Temperature", f"{settings.temperature:g}")
    with right:
        st.metric("Chunk size / overlap", f"{settings.chunk_size} / {settings.chunk_overlap}")
        st.metric("Max upload", f"{settings.max_upload_mb} MB")
        st.metric("API key", settings.masked_key)

    st.subheader("Build progress")
    st.table(
        {
            "Phase": [p for p, _, _ in PHASES],
            "Scope": [s for _, s, _ in PHASES],
            "Status": [x for _, _, x in PHASES],
        }
    )


if __name__ == "__main__":
    main()
