# ASME-Style RAG Evaluation Harness

Retrieval evaluation harness for engineering standards documents. Built as a concrete
demonstration of the retrieval-evaluation workstream from the DeepMechanix JD.

Uses NASA-STD-5019 and API-510 as public proxies for ASME BPVC (licensed). Same
architecture, no code changes needed to swap in ASME.

Full design story — decisions, failures, and fixes — in `deep-mechanix-journey.docx`.

---

## Results

| Metric                      | Value |
|-----------------------------|-------|
| In-corpus Recall@5          | 1.000 |
| In-corpus MRR               | 0.944 |
| Abstention Precision        | 1.000 |
| Abstention Recall           | 0.750 |
| Citation Traversal Hit Rate | 1.000 |

20 queries: 12 in-corpus, 5 adversarial out-of-corpus, 3 ambiguous.
Run `python -m src.eval` to reproduce.

---

## How It Works

**Clause-aware chunking** — each numbered clause is one chunk. Splitting a clause
across a token window boundary would be the worst failure mode here.

**Hybrid retrieval** — BM25 + dense (bge-large-en-v1.5 → FAISS) fused via RRF (k=60).

**Dual abstention gate** — abstain if cosine similarity < 0.40 OR max BM25 score < 10.0.
Both conditions required independently because bge-large maps all English text into a
high-similarity region, so the cosine gate alone fails on off-domain queries. BM25 has
no such problem — a paracetamol query scores ~2 against engineering standards.

**Citation graph** — cross-references stored in Neo4j AuraDB. Referenced clauses
appended to the result set at query time. Without this, retrieving UG-27 without
Section II Part D Table 1A gives you half a formula.

**Prefix matching for recall** — `"4.1"` matches `"4.1.1"`, `"4.1.2"`, `"4.1#part0"`.
Strict equality would penalize retrieving the most specific relevant sub-clause.

---

## Run It

```bash
pip install pymupdf sentence-transformers faiss-cpu rank-bm25 pydantic \
    neo4j pyyaml python-dotenv

# place nasa_std_5019.pdf and api_510.pdf in data/raw/

cp .env.example .env        # add Neo4j credentials

python -m src.parse         # PDF -> clause chunks -> SQLite
python -m src.embed         # embeddings -> FAISS + BM25
python -m src.graph build   # citation graph -> Neo4j

python -m src.eval --debug  # run 20 queries, write eval/reports/
```

---

## Stack

Python 3.11 · PyMuPDF · bge-large-en-v1.5 · FAISS IndexFlatIP · rank-bm25 ·
RRF fusion · SQLite · Neo4j AuraDB Free · Pydantic

Cross-encoder reranker scaffold included in `retrieve.py` (`use_reranker=False`).
Adds ~4pt MRR; disabled to keep cold-start < 5s.
