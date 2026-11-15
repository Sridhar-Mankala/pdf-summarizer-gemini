# PDF Summarizer + RAG (Gemini · LangChain · Streamlit)

Upload a PDF, get a structured summary, then ask questions about it with page citations.
Built with LangChain, Google Gemini and Streamlit.

> **Status:** Phase 0 complete (environment, API key and GitHub auth validated).
> The app itself lands in Phases 1-5; see [Roadmap](#roadmap).

## Features (target)

- **Structured summaries** - TL;DR, key points, section-wise notes, action items
- **Long-document safe** - stuffed prompt for short PDFs, map-reduce for 40+ pages
- **Ask your PDF** - RAG over the document with page-level citations
- **Streamlit UI** - upload, streaming output, Markdown download, chat tab
- **No secrets in git** - keys live in `.env`, never in code or commits

## Quick start

```powershell
# 1. dependencies (uses the existing .venv)
.\.venv\Scripts\pip.exe install -r requirements.txt

# 2. credentials
copy .env.example .env
#    then set GEMINI_API_KEY in .env  ->  https://aistudio.google.com/apikey

# 3. verify everything before running the app
.\.venv\Scripts\python.exe scripts/check_setup.py

# 4. run (available from Phase 4)
.\.venv\Scripts\streamlit.exe run app.py
```

## Roadmap

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Tooling, `.env`, live Gemini key check, GitHub login | done |
| 1 | Project skeleton, pinned deps, config layer | planned |
| 2 | PDF ingestion, text cleaning, token-aware chunking | planned |
| 3 | Summarization engine (stuff + map-reduce, retries) | planned |
| 4 | Streamlit UI - upload, stream, export | planned |
| 5 | RAG Q&A with Gemini embeddings + Chroma citations | planned |
| 6 | Tests, lint, CI, docs | planned |
| 7 | Release `v0.1.0` | planned |

## Project layout

```
app.py                  # Streamlit entry point
scripts/check_setup.py  # Phase 0 gate: Gemini key + git + gh auth
src/pdf_summarizer/     # config, loaders, chunking, prompts, summarize, rag
tests/                  # pytest suite (LLM calls mocked)
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `HTTP 400/403` from Gemini | key is wrong or the Generative Language API is not enabled for the project |
| `HTTP 429` | free-tier quota hit; wait or raise `CHUNK_SIZE` to send fewer requests |
| "no extractable text" | scanned/image PDF - OCR is out of scope |
| `gh auth status` fails | `gh auth login -h github.com -w`, then `gh auth refresh -s repo` |

## License

MIT
