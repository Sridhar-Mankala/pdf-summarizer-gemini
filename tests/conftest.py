from __future__ import annotations

from collections.abc import Callable

import pytest


def _build_pdf(page_texts: list[str], font_size: int = 18) -> bytes:
    """Hand-roll a minimal, valid PDF so tests need no binary fixtures."""
    page_count = len(page_texts)
    font_id = 3 + 2 * page_count
    kids = " ".join(f"{3 + i} 0 R" for i in range(page_count))

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode(),
    }

    for i, text in enumerate(page_texts):
        page_id = 3 + i
        content_id = 3 + page_count + i
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        ).encode()
        escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        stream = f"BT /F1 {font_size} Tf 72 720 Td ({escaped}) Tj ET".encode()
        objects[content_id] = (
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )

    objects[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for object_id in range(1, font_id + 1):
        offsets[object_id] = len(out)
        out += f"{object_id} 0 obj\n".encode() + objects[object_id] + b"\nendobj\n"

    xref_offset = len(out)
    size = font_id + 1
    out += f"xref\n0 {size}\n".encode()
    out += b"0000000000 65535 f \n"
    for object_id in range(1, size):
        out += f"{offsets[object_id]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    return bytes(out)


@pytest.fixture
def make_pdf() -> Callable[..., bytes]:
    return _build_pdf


@pytest.fixture
def sample_pdf_bytes(make_pdf: Callable[..., bytes]) -> bytes:
    return make_pdf(
        [
            "Retrieval augmented generation combines search with language models. "
            "It grounds answers in retrieved evidence.",
            "Chunking strategy matters. Page level chunking keeps citations exact, "
            "while overlap preserves context across boundaries.",
            "Evaluation should check faithfulness and citation accuracy, not just fluency.",
        ]
    )


@pytest.fixture
def blank_pdf_bytes(make_pdf: Callable[..., bytes]) -> bytes:
    return make_pdf(["", ""])
