from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document: str
    section: str
    clause: str
    paragraph: str | None = None
    page: int
    chunk_type: Literal["prose", "table", "formula"]
    content: str
    table_data: dict | None = None
    references: list[str] = Field(default_factory=list)
    token_count: int = 0


class RetrievalResult(BaseModel):
    query: str
    chunks: list[Chunk]
    scores: list[float]
    abstain: bool
    top_score: float
    retrieval_method: str = "hybrid_rrf"


class EvalQuery(BaseModel):
    id: str
    category: Literal[
        "in_corpus_direct",
        "in_corpus_cross_reference",
        "out_of_corpus_adversarial",
        "ambiguous",
    ]
    query: str
    expected_clauses: list[str] = Field(default_factory=list)
    expected_documents: list[str] = Field(default_factory=list)
    expected_referenced_clauses: list[str] = Field(default_factory=list)
    expect_abstain: bool = False
    note: str | None = None


class EvalResult(BaseModel):
    query_id: str
    category: str
    query: str
    retrieved_clause_ids: list[str]
    expected_clauses: list[str]
    abstained: bool
    expected_abstain: bool
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    citation_hit: bool | None = None
    correct: bool


class EvalReport(BaseModel):
    timestamp: str
    total_queries: int
    in_corpus_recall_at_1: float
    in_corpus_recall_at_3: float
    in_corpus_recall_at_5: float
    in_corpus_recall_at_10: float
    in_corpus_mrr: float
    abstention_precision: float
    abstention_recall: float
    citation_traversal_hit_rate: float
    per_category: dict[str, dict[str, float]]
    results: list[EvalResult]
