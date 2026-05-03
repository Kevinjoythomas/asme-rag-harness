"""Embedding + indexing.

Dense: BAAI/bge-large-en-v1.5 via sentence-transformers → FAISS IndexFlatIP
       (inner product on L2-normalised vectors = cosine similarity).
Sparse: BM25Okapi via rank-bm25, pickled in-process.
Storage: SQLite for chunk content + metadata.
Citations: Neo4j AuraDB (optional) for the clause cross-reference graph.

Considered text-embedding-3-large (better, paid) and bge-m3 (multilingual,
overkill for English-only engineering standards). bge-large-en-v1.5 is the
sweet spot: free, strong on technical text, ~1.3 GB download.
"""
from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.schema import Chunk

MODEL_NAME = "BAAI/bge-large-en-v1.5"
_ROOT = Path(__file__).parent.parent
DB_PATH = _ROOT / "data/index.db"
FAISS_PATH = _ROOT / "data/faiss.index"
BM25_PATH = _ROOT / "data/bm25.pkl"
META_PATH = _ROOT / "data/index_meta.json"


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_index(chunks: list[Chunk]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --- SQLite ---
    con = sqlite3.connect(str(DB_PATH))
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id   TEXT PRIMARY KEY,
            document   TEXT,
            section    TEXT,
            clause     TEXT,
            paragraph  TEXT,
            page       INTEGER,
            chunk_type TEXT,
            content    TEXT,
            table_data  TEXT,
            refs        TEXT,
            token_count INTEGER
        )
    """)
    con.executemany(
        "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                c.chunk_id, c.document, c.section, c.clause,
                c.paragraph, c.page, c.chunk_type, c.content,
                json.dumps(c.table_data), json.dumps(c.references), c.token_count,
            )
            for c in chunks
        ],
    )
    con.commit()
    con.close()
    print(f"SQLite: {len(chunks)} chunks stored")

    # --- Dense embeddings ---
    model = SentenceTransformer(MODEL_NAME)
    texts = [c.content for c in chunks]
    print(f"Embedding {len(texts)} chunks with {MODEL_NAME}...")
    embeddings: np.ndarray = model.encode(
        texts, show_progress_bar=True, normalize_embeddings=True, batch_size=32
    )

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(FAISS_PATH))
    print(f"FAISS: {index.ntotal} vectors, dim={dim}")

    # --- BM25 ---
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    chunk_ids = [c.chunk_id for c in chunks]
    BM25_PATH.write_bytes(pickle.dumps({"bm25": bm25, "chunk_ids": chunk_ids}))
    print(f"BM25: {len(chunks)} documents indexed")

    # --- Position metadata (FAISS position i → chunk_ids[i]) ---
    META_PATH.write_text(
        json.dumps({"model": MODEL_NAME, "chunk_ids": chunk_ids, "total": len(chunks)}),
        encoding="utf-8",
    )

    # --- Neo4j citation graph (optional) ---
    from src.graph import build_graph, graph_available

    if graph_available():
        build_graph(chunks)
    else:
        print("Neo4j: skipped (NEO4J_URI not set — set it in .env to enable citation graph)")


def load_chunks_from_db(db_path: Path = DB_PATH) -> list[Chunk]:
    con = sqlite3.connect(str(db_path))
    rows = con.execute("SELECT * FROM chunks ORDER BY rowid").fetchall()
    con.close()
    return [
        Chunk(
            chunk_id=r[0], document=r[1], section=r[2], clause=r[3],
            paragraph=r[4], page=r[5], chunk_type=r[6], content=r[7],
            table_data=json.loads(r[8]) if r[8] and r[8] != "null" else None,
            references=json.loads(r[9]) if r[9] else [],
            token_count=r[10] or 0,
        )
        for r in rows
    ]


def main(parsed_dir: Path = Path("data/parsed")) -> None:
    chunks_path = parsed_dir / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Run chunk.py first: {chunks_path}")
    raw = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = [Chunk(**c) for c in raw]
    build_index(chunks)


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    pd = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/parsed")
    main(pd)
