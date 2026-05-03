"""Clause-aware chunking.

Clauses are the unit of authority in engineering standards — splitting a clause
across chunk boundaries would be the worst possible failure mode (you'd retrieve
half a rule). This module groups parsed blocks by clause ID, splits only when a
clause exceeds MAX_TOKENS, and extracts cross-references for the citation graph.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.schema import Chunk

MAX_TOKENS = 2000
_CHARS_PER_TOKEN = 4  # conservative approximation

# Cross-reference patterns for numeric engineering standards
_REF_PATTERNS = [
    re.compile(r"\b(\d{1,2}(?:\.\d{1,3}){1,4})\b"),      # 4.1.2.3
    re.compile(r"\bSection\s+(\d{1,2}(?:\.\d{1,3})*)\b"),  # Section 4.1
    re.compile(r"\bTable\s+(\d{1,2}(?:[.-]\d{1,3})*)\b"),  # Table 4-1 or Table 4.1
    re.compile(r"\bFigure\s+(\d{1,2}(?:[.-]\d{1,3})*)\b"), # Figure 3-2
    re.compile(r"\bAppendix\s+([A-Z])\b"),                  # Appendix A
]


def _tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def extract_references(text: str) -> list[str]:
    refs: set[str] = set()
    for pat in _REF_PATTERNS:
        for m in pat.finditer(text):
            refs.add(m.group(1))
    return sorted(refs)


def _split_long_clause(
    clause_id: str,
    section: str,
    document: str,
    blocks: list[dict],
) -> list[Chunk]:
    """Split a clause that exceeds MAX_TOKENS on paragraph boundaries."""
    chunks: list[Chunk] = []
    current: list[str] = []
    current_page = blocks[0]["page"]
    part = 0

    def flush() -> None:
        nonlocal part, current_page
        if not current:
            return
        content = "\n".join(current)
        chunks.append(
            Chunk(
                document=document,
                section=section,
                clause=f"{clause_id}#part{part}",
                paragraph=None,
                page=current_page,
                chunk_type="prose",
                content=content,
                references=extract_references(content),
                token_count=_tokens(content),
            )
        )
        part += 1
        current.clear()

    for block in blocks:
        text = block["text"]
        if _tokens("\n".join(current) + "\n" + text) > MAX_TOKENS and current:
            flush()
            current_page = block["page"]
        current.append(text)

    flush()
    return chunks


def build_chunks(parsed_blocks: list[dict], document: str) -> list[Chunk]:
    # Group blocks by clause, preserving insertion order
    groups: dict[str, list[dict]] = {}
    for block in parsed_blocks:
        cid = block["clause"]
        groups.setdefault(cid, []).append(block)

    chunks: list[Chunk] = []
    for cid, blocks in groups.items():
        section = blocks[0]["section"]
        page = blocks[0]["page"]
        content = "\n".join(b["text"] for b in blocks)

        if _tokens(content) > MAX_TOKENS:
            chunks.extend(_split_long_clause(cid, section, document, blocks))
        else:
            chunks.append(
                Chunk(
                    document=document,
                    section=section,
                    clause=cid,
                    paragraph=None,
                    page=page,
                    chunk_type="prose",
                    content=content,
                    references=extract_references(content),
                    token_count=_tokens(content),
                )
            )

    return chunks


def load_and_chunk(parsed_path: Path) -> list[Chunk]:
    blocks: list[dict] = json.loads(parsed_path.read_text(encoding="utf-8"))
    return build_chunks(blocks, parsed_path.stem)


def main(parsed_dir: Path) -> None:
    all_chunks: list[Chunk] = []
    for json_file in sorted(parsed_dir.glob("*.json")):
        if json_file.name == "chunks.json":
            continue
        chunks = load_and_chunk(json_file)
        all_chunks.extend(chunks)
        print(f"{json_file.stem}: {len(chunks)} chunks")

    out = parsed_dir / "chunks.json"
    out.write_text(
        json.dumps([c.model_dump() for c in all_chunks], indent=2),
        encoding="utf-8",
    )
    print(f"Total: {len(all_chunks)} chunks → {out}")


if __name__ == "__main__":
    import sys

    pd = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/parsed")
    main(pd)
