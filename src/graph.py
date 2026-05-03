"""Neo4j AuraDB citation graph.

Clause nodes and REFERENCES edges are the backbone of cross-reference traversal.
A clause that cites another clause gets an edge; at query time we can walk one hop
to pull in all dependencies for a given result set.

Setup: create a free AuraDB instance at neo4j.com/cloud/aura-free, copy the
connection URI and password into .env, then `python -m src.graph build` to populate.
"""
from __future__ import annotations

import os
from typing import Any

from src.schema import Chunk

_driver: Any = None  # lazy-loaded


def _get_driver() -> Any:
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase  # type: ignore[import-untyped]

        uri = os.environ.get("NEO4J_URI", "")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "")
        if not uri:
            raise EnvironmentError("NEO4J_URI not set. See .env.example.")
        _driver = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


def close() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


# ─────────────────────────── schema ─────────────────────────────
# (:Clause {clause_id, document, chunk_id, page})
# (:Clause)-[:REFERENCES]->(:Clause)

def build_graph(chunks: list[Chunk]) -> None:
    """Write all clause nodes and cross-reference edges to Neo4j."""
    driver = _get_driver()
    clause_set = {c.clause for c in chunks}

    with driver.session() as session:
        # Upsert nodes
        session.run("CREATE INDEX clause_id IF NOT EXISTS FOR (n:Clause) ON (n.clause_id)")
        for c in chunks:
            session.run(
                """
                MERGE (n:Clause {clause_id: $cid, document: $doc})
                SET n.chunk_id = $chunk_id, n.page = $page
                """,
                cid=c.clause, doc=c.document, chunk_id=c.chunk_id, page=c.page,
            )

        # Write directed REFERENCES edges
        edge_count = 0
        for c in chunks:
            for ref in c.references:
                if ref in clause_set:
                    session.run(
                        """
                        MATCH (a:Clause {clause_id: $from_id, document: $doc})
                        MATCH (b:Clause {clause_id: $to_id})
                        MERGE (a)-[:REFERENCES]->(b)
                        """,
                        from_id=c.clause, to_id=ref, doc=c.document,
                    )
                    edge_count += 1

    node_count = len(chunks)
    print(f"Neo4j: {node_count} clause nodes, {edge_count} REFERENCES edges")


def get_referenced_chunk_ids(clause_id: str, document: str) -> list[str]:
    """Return chunk_ids one hop away from the given clause node."""
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (a:Clause {clause_id: $cid, document: $doc})-[:REFERENCES]->(b:Clause)
            RETURN b.chunk_id AS chunk_id
            """,
            cid=clause_id, doc=document,
        )
        return [r["chunk_id"] for r in result if r["chunk_id"]]


def graph_available() -> bool:
    """Return True if NEO4J_URI is configured."""
    return bool(os.environ.get("NEO4J_URI", ""))


def main(action: str = "build") -> None:
    import json
    from pathlib import Path

    from dotenv import load_dotenv

    from src.schema import Chunk

    load_dotenv()

    if action == "build":
        chunks_path = Path("data/parsed/chunks.json")
        if not chunks_path.exists():
            raise FileNotFoundError("Run chunk.py first")
        chunks = [Chunk(**c) for c in json.loads(chunks_path.read_text(encoding="utf-8"))]
        build_graph(chunks)
        close()
    elif action == "stats":
        driver = _get_driver()
        with driver.session() as session:
            n = session.run("MATCH (n:Clause) RETURN count(n) AS n").single()["n"]
            e = session.run("MATCH ()-[r:REFERENCES]->() RETURN count(r) AS e").single()["e"]
        print(f"Nodes: {n}  Edges: {e}")
        close()
    else:
        print(f"Unknown action: {action}. Use 'build' or 'stats'.")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "build")
