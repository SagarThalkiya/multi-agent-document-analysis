"""
Utility helpers to load document contents into plain text.
"""

from pathlib import Path
from typing import Union

import pdfplumber


class UnsupportedDocumentError(ValueError):
    """Raised when we cannot parse the given document type."""


def extract_text(file_path: Union[str, Path]) -> str:
    """
    Return the textual content of a TXT or PDF file.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found at {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return _extract_pdf_text(path)

    raise UnsupportedDocumentError(
        f"Unsupported document extension '{suffix}'. Only PDF and TXT are allowed."
    )


def _extract_pdf_text(path: Path) -> str:
    """
    Use pdfplumber to pull text from each page.
    """

    text_parts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text.strip())
    return "\n".join(part for part in text_parts if part)

