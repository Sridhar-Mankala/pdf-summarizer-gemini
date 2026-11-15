# PDF Summarizer + RAG

Upload a PDF, get a grounded structured summary, then ask questions about it with page-level citations.

Built with **LangChain**, **Google Gemini** and **Streamlit**.

[![CI](https://github.com/Sridhar-Mankala/pdf-summarizer-gemini/actions/workflows/ci.yml/badge.svg)](https://github.com/Sridhar-Mankala/pdf-summarizer-gemini/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## What it does

- **Structured summaries** - TL;DR, key points, section notes, action items/numbers, open questions
- **Long-document safe** - one prompt for short PDFs, map-reduce for documents over ~60k tokens
- **Ask your PDF** - retrieval-augmented Q&A over Gemini embeddings, with page citations and an explicit refusal when the answer is not in the document
- **Markdown export** - download the summary as a `.md` file
- **Page inspector** - read the extracted text per page to verify what the model saw
- **Fail-fast setup** - one script validates the API key, git and GitHub auth before anything else

## Architecture

```
PDF upload
   │
   ▼
pypdf loader ──► per-page text + 1-based page numbers      (loaders.py)
   │
   ▼
recursive chunker (chars, page-tagged)                     (chunking.py)
   │
   ├──► token budget ──► stuff prompt ──────────────┐
   │                    (<= 60k tokens)             │
   │                                                ├──► Gemini ──► Markdown summary
   └──► map-reduce: batches of ~12k tokens ─────────┘                (summarize.py)
        map notes ──► hierarchical merge ──► final reduce

chunks ──► Gemini embeddings ──► Chroma (in memory) ──► top-k retrieval
                                                          │
                                          grounded answer + [p. N] citations
                                                          (rag.py)
```

Design choices worth knowing:

- **Page-tagged chunks** keep citations exact; the trade-off is a little lost cross-page context.
- **Strategy is chosen by tokens, not page count** - a dense 6-page PDF can outgrow a light 40-page one.
- **The model is told to refuse** rather than guess, and refusals carry no citations.
- **Indexes are per document** and keyed by the file's SHA-256, so two uploads never share chunks.
- **All model calls are injectable**, so the test suite runs offline with fake models.

## Quick start

Requires Python 3.12 (3.10+ should work) and a Gemini API key.

```powershell
git clone https://github.com/Sridhar-Mankala/pdf-summarizer-gemini.git
cd pdf-summarizer-gemini

python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env     # then set GEMINI_API_KEY
python scripts/check_setup.py
streamlit run app.py
```

Get a free key at <https://aistudio.google.com/apikey>.

On macOS/Linux use `source .venv/bin/activate` and `cp .env.example .env`.

## Configuration

`.env` is git-ignored. Only `.env.example` is committed.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | - | **Required.** Google AI Studio key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Chat model for summaries and answers |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Embeddings for the RAG index |
| `TEMPERATURE` | `0.2` | Low keeps summaries factual |
| `MAX_UPLOAD_MB` | `25` | Upload size guard |
| `CHUNK_SIZE` | `4000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Characters of overlap |
| `LOG_LEVEL` | `INFO` | App log verbosity |

Model and temperature are also adjustable in the sidebar without editing `.env`.

## Verify the setup

```powershell
python scripts/check_setup.py
```

Checks Python, the virtualenv, `.env`, a **live Gemini call**, the git identity and GitHub CLI auth. Exit code 0 only when everything passes.

## Tests

```powershell
pytest
ruff check .
ruff format --check .
```

53 tests, no API key required: PDFs are generated in memory, models are LangChain fakes, and embeddings are deterministic.

## Project layout

```
app.py                      Streamlit UI (summary, ask, pages tabs)
scripts/check_setup.py      Phase-0 gate: key + git + GitHub login
src/pdf_summarizer/
  config.py                 pydantic-settings, validated once, cached
  loaders.py                pypdf -> page-numbered, cleaned text
  chunking.py               page-aware recursive splitting
  prompts.py                summary, map/reduce and RAG prompts
  summarize.py              stuff vs map-reduce engine
  rag.py                    Chroma index + citation-aware Q&A
  llm.py                    Gemini chat / embeddings factories
  utils.py                  errors, logging, hashing, token estimates
tests/                      pytest suite (offline)
```

## Limitations

- **Scanned PDFs are not supported** - there is no OCR; you get a clear error instead of an empty summary.
- **Password-protected PDFs** must be unlocked first (empty-password PDFs are handled).
- **Free-tier quota**: map-reduce on a long PDF makes many calls. If you hit `429`, wait, or raise `CHUNK_SIZE` to send fewer, larger requests.
- **Token estimates are heuristic** (~4 chars per token), for budgeting only - not billing.
- **Answers are only as good as the extraction**; use the Pages tab to check what the parser saw.

## Roadmap

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Tooling, `.env`, live key check, GitHub auth | done |
| 1 | Skeleton, pinned deps, config layer | done |
| 2 | PDF ingestion, cleaning, page-aware chunking | done |
| 3 | Summarization engine (stuff + map-reduce) | done |
| 4 | Streamlit UI with digest-keyed state | done |
| 5 | RAG Q&A with page citations | done |
| 6 | Tests, lint, CI, docs | done |
| 7 | `v0.1.0` release | done |

## License

MIT - see [LICENSE](LICENSE).
