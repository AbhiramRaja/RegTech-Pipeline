"""
src/ingestion/pdf_parser.py

Section-aware PDF text and table extractor using PyMuPDF.
Returns a list of clause dicts: {clause_id, text, page, section}.

Design notes:
  - Sections are detected by heuristic: text blocks in bold/large font, or
    lines that look like headings (ALL CAPS, numbered, short length).
  - Empty or malformed PDFs are handled gracefully — returns [] with a warning.
  - Each chunk gets a deterministic clause_id: "{doc_id}::{page}::{idx}".
"""

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Try importing PyMuPDF; alias as fitz
try:
    import fitz  # PyMuPDF
except ImportError as e:
    raise ImportError("PyMuPDF is required. Run: pip install pymupdf") from e


def _looks_like_heading(text: str) -> bool:
    """Heuristic: short lines that are all-caps or numbered like '1.' / 'I.'"""
    stripped = text.strip()
    if not stripped or len(stripped) > 120:
        return False
    # Numbered headings: "1.", "1.1", "Section 1", "ARTICLE I"
    if re.match(r"^(section\s+\d+|article\s+[ivxlcdm]+|\d+\.(\d+\.?)*)\s", stripped, re.IGNORECASE):
        return True
    # Short all-caps (likely a heading)
    words = stripped.split()
    if len(words) <= 8 and stripped == stripped.upper() and any(c.isalpha() for c in stripped):
        return True
    return False


def extract_clauses(pdf_path: str | Path, doc_id: Optional[str] = None) -> list[dict]:
    """
    Parse a PDF and return a list of clause dicts.

    Args:
        pdf_path: Path to the PDF file.
        doc_id:   Identifier for this document. Defaults to the file stem.

    Returns:
        List of dicts with keys: clause_id, text, page, section.
        Returns [] (with a logged warning) on any parse error.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.warning("PDF not found: %s", pdf_path)
        return []

    doc_id = doc_id or pdf_path.stem

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.warning("Could not open PDF %s: %s", pdf_path, exc)
        return []

    if doc.page_count == 0:
        logger.warning("PDF has no pages: %s", pdf_path)
        return []

    clauses: list[dict] = []
    current_section = "preamble"
    clause_idx = 0

    for page_num, page in enumerate(doc, start=1):
        try:
            blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
        except Exception as exc:
            logger.warning("Failed to read page %d of %s: %s", page_num, pdf_path, exc)
            continue

        for block in blocks:
            # block_type 0 = text, 1 = image
            if len(block) < 6 or block[6] != 0:
                continue

            raw_text: str = block[4]
            text = raw_text.strip()
            if not text:
                continue

            # Detect section headings
            if _looks_like_heading(text):
                current_section = text[:200]  # cap section name length
                continue

            # Skip very short noise (page numbers, headers/footers)
            if len(text) < 20:
                continue

            clause_id = f"{doc_id}::p{page_num}::c{clause_idx}"
            clauses.append(
                {
                    "clause_id": clause_id,
                    "text": text,
                    "page": page_num,
                    "section": current_section,
                }
            )
            clause_idx += 1

    doc.close()
    logger.info("Parsed %d clauses from '%s'", len(clauses), pdf_path.name)
    return clauses


def extract_clauses_from_dir(directory: str | Path, source_type: str = "unknown") -> list[dict]:
    """
    Parse every PDF in a directory and return all clauses.
    Adds a 'source_type' key to each clause dict.
    """
    directory = Path(directory)
    all_clauses: list[dict] = []
    for pdf_file in sorted(directory.glob("*.pdf")):
        clauses = extract_clauses(pdf_file)
        for c in clauses:
            c["source_type"] = source_type
        all_clauses.extend(clauses)
    return all_clauses
