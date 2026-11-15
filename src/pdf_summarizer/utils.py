"""Shared helpers: error types, logging, hashing and rough token maths."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable

LOGGER_NAME = "pdf_summarizer"


class AppError(Exception):
    """Base class for errors that are safe to show to a user."""


class PdfError(AppError):
    """The uploaded file could not be read as a PDF."""


class ScannedPdfError(PdfError):
    """No extractable text - the PDF is probably a scan or images."""


class EmptyDocumentError(AppError):
    """The document has no usable text after cleaning."""


def configure_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger


def get_logger(component: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{component}")


_WHITESPACE = re.compile(r"[ \t\u00a0]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str) -> str:
    """Normalise PDF text without destroying paragraph structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # re-join words split across lines
    text = _WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def estimate_tokens(text: str) -> int:
    """Cheap heuristic (~4 chars/token). Good enough for budgeting, not billing."""
    return max(1, len(text) // 4) if text else 0


def estimate_tokens_many(texts: Iterable[str]) -> int:
    return sum(estimate_tokens(t) for t in texts)


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"
