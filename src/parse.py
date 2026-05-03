"""PDF → structured blocks using PyMuPDF.

Two-pass approach: walk each page's text blocks, detect clause headings via
regex, and tag every span with the current clause context.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import fitz  # pymupdf


# Engineering-standards clause patterns (numeric hierarchy)
_CLAUSE_RE = re.compile(
    r"^(\d{1,2}(?:\.\d{1,3}){0,4})\s*(?:\s+|$)"
)

ParsedBlock = dict  # {page, clause, section, text, is_heading}


def _detect_clause(line: str) -> str | None:
    """Return clause ID if line opens a new clause, else None."""
    m = _CLAUSE_RE.match(line.strip())
    if not m:
        return None
    clause = m.group(1)
    # Reject bare integers that are just list numbers (e.g. "1. Introduction" with
    # no sub-level) — only accept if the line has meaningful text after the number
    rest = line.strip()[len(clause):].strip()
    if not rest and "." not in clause:
        return None
    return clause


def _infer_section(clause: str) -> str:
    return clause.split(".")[0]


def parse_pdf(pdf_path: Path) -> list[ParsedBlock]:
    doc = fitz.open(str(pdf_path))
    blocks: list[ParsedBlock] = []
    current_clause = "preamble"
    current_section = "0"

    for page_num, page in enumerate(doc, start=1):
        raw_blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
        for raw in raw_blocks:
            if raw[6] != 0:  # skip image blocks
                continue
            for line in raw[4].split("\n"):
                line = line.strip()
                if not line:
                    continue

                detected = _detect_clause(line)
                if detected:
                    current_clause = detected
                    current_section = _infer_section(current_clause)

                blocks.append(
                    {
                        "page": page_num,
                        "clause": current_clause,
                        "section": current_section,
                        "text": line,
                        "is_heading": detected is not None,
                    }
                )

    doc.close()
    return blocks


def save_parsed(blocks: list[ParsedBlock], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(blocks, indent=2), encoding="utf-8")


def main(raw_dir: Path, parsed_dir: Path) -> None:
    parsed_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {raw_dir}. See data/raw/README.md for download instructions.")
        return
    for pdf in pdfs:
        print(f"Parsing {pdf.name}...")
        blocks = parse_pdf(pdf)
        out = parsed_dir / f"{pdf.stem}.json"
        save_parsed(blocks, out)
        print(f"  → {len(blocks)} blocks → {out}")


if __name__ == "__main__":
    import sys

    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    parsed = Path("data/parsed")
    main(raw, parsed)
