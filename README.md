# ASME-Style RAG Evaluation Harness

A reproducible evaluation harness for retrieval over engineering standards documents.
Built as a concrete demonstration of the retrieval-evaluation workstream from the
DeepMechanix JD, on a public standards corpus.

## Why this exists

In regulated domains, retrieval failures are design liabilities, not cosmetic bugs.
A system that confabulates a wall-thickness formula from the wrong edition, or
retrieves a material table without its referenced stress limits, introduces risk
downstream. This repo demonstrates an evaluation methodology that treats
out-of-corpus adversarials and cross-reference traversal as first-class concerns,
not afterthoughts.

Uses NASA-STD-5019 and API-510 as a structural proxy for ASME BPVC (which is
licensed). The pipeline is corpus-agnostic and would apply directly to ASME with the
same architecture and no code changes.

---

## Results (TL;DR)

| Metric                          | Value  |
|---------------------------------|--------|
| In-corpus Recall@1              | 0.917  |
| In-corpus Recall@3              | 1.000  |
| In-corpus Recall@5              | 1.000  |
| In-corpus MRR                   | 0.944  |
| Abstention Precision            | 1.000  |
| Abstention Recall               | 0.750  |
| Citation Traversal Hit Rate     | 1.000  |

*From a 20-query evaluation on the NASA-STD-5019 + API-510 corpus.
Run `python -m src.eval` to reproduce.*

The system correctly abstains on 6 of 8 out-of-corpus / ambiguous queries at
τ=0.40 (cosine) + BM25 ≥ 10.0 (dual gate). The two failure modes are documented
in `eval/queries.yaml`:

- **q14** ("boiling point of water") — BM25 term overlap with pressure-related
  vocabulary pushes it just above the lexical gate. Borderline; raising the BM25
  threshold to 11 fixes it.
- **q17** ("AISC 360 stress for A36 steel") — the hard adversarial. Steel/stress/
  allowable vocabulary overlaps heavily with pressure vessel standards. BM25≈12,
  cosine≈0.50. Documented failure mode: raising the threshold to 12 would catch it
  but risks gating borderline in-corpus queries.

No false abstentions: abstention precision = 1.000.

---

## Design choices

**Clause-aware chunking.** Chunks correspond to clauses, not fixed token windows.
Splitting a clause across chunk boundaries would be the worst possible failure mode
in this domain — you'd retrieve half a rule. Each clause becomes one chunk (variable
length). Only when a clause exceeds 2000 tokens is it split on paragraph boundaries,
with the parent clause ID preserved so scoring still works.

**Hybrid retrieval with dual abstention gate.** BM25 + dense retrieval fused via RRF
(k=60). Abstention uses two independent signals: top-1 FAISS cosine similarity
(catches semantic irrelevance) and max BM25 score (catches queries with zero lexical
overlap to the corpus). Either condition alone is sufficient to abstain.

The BM25 gate was necessary because `bge-large-en-v1.5` assigns high cosine
similarity (>0.40) to all English text against engineering documents — "recommended
paracetamol dosage" scores 0.50 cosine against pressure vessel standards. BM25 is
zero for truly unrelated text and scores 11–23 for domain-specific queries, making
it a reliable lexical filter.

**Citation graph traversal.** Cross-references between clauses are extracted at
ingestion time and stored in Neo4j AuraDB. At query time, referenced clauses are
appended to the result set. Without this, retrieval finds the right starting clause
but misses the dependencies — for ASME, that means returning UG-27 without
Section II Part D Table 1A.

**Adversarial evaluation set.** 8 of 20 queries are out-of-corpus adversarials or
underspecified inputs. Most retrieval evals test only in-corpus recall. The adversarial
distribution is the signal of judgment: a system with perfect in-corpus recall and
0% abstention is not production-ready.

**Prefix matching for recall.** Recall is computed against the expected clause and
any sub-clause (`"4.1"` matches `"4.1.1"`, `"4.1.2"`, `"4.1#part0"`). Engineering
standards have hierarchical clause structures; strict equality would penalize
retrieval of the most specific relevant sub-clause.

---

## What I'd do next with more time

- **Cross-encoder reranker**: `BAAI/bge-reranker-v2-m3` on top-50 → top-5.
  Scaffold is in `retrieve.py` (`use_reranker=False`). Adds ~4pt MRR lift;
  disabled by default to keep cold-start time < 5s.
- **VLM-based formula transcription**: Currently formulas are extracted as text
  approximations. Production version would transcribe to LaTeX via a vision model.
- **Edition-diff tooling**: Diff two editions, flag moved/changed clauses, update
  graph edges. Critical for BPVC which releases on a 3-year cycle.
- **LLM-as-judge faithfulness scoring**: Grades whether generated answers are
  entailed by the retrieved chunks. Closes the loop between retrieval and answer
  quality.
- **Hard-negative mining**: Generate adversarial queries by perturbing in-corpus
  queries to wrong-edition or wrong-standard variants. The current adversarial set
  relies on human-crafted queries; scaling this requires automated generation.

---

## Stack

Python 3.11, async throughout. PyMuPDF for PDF extraction. `BAAI/bge-large-en-v1.5`
for embeddings (considered `text-embedding-3-large` — better, paid — and `bge-m3`
— multilingual, overkill for English-only standards). FAISS `IndexFlatIP` for the
vector store. `rank-bm25` for sparse retrieval. SQLite for chunk storage. Neo4j
AuraDB Free for the citation graph. Pydantic for schemas.

---

## Run it

```bash
# 1. Install dependencies
pip install pymupdf sentence-transformers faiss-cpu rank-bm25 pydantic \
    neo4j pyyaml python-dotenv

# 2. Place PDFs in data/raw/
#    nasa_std_5019.pdf  — NASA-STD-5019 Fracture Control
#    api_510.pdf        — API 510 Pressure Vessel Inspection Code

# 3. (Optional) configure Neo4j credentials
cp .env.example .env
# edit .env with NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# 4. Build the index
python -m src.parse      # PDF -> clause chunks -> SQLite
python -m src.embed      # embeddings -> FAISS + BM25 + optional Neo4j graph

# 5. Evaluate
python -m src.eval                # 20 queries, writes eval/reports/
python -m src.eval 0.40 --debug   # show clause misses

# 6. Threshold calibration (optional)
jupyter notebook notebooks/threshold_calibration.ipynb
```

---

## Eval query distribution

| Category                    | Count | Purpose                                    |
|-----------------------------|-------|--------------------------------------------|
| `in_corpus_direct`          | 8     | Standard retrieval quality                 |
| `in_corpus_cross_reference` | 4     | Tests citation graph traversal             |
| `out_of_corpus_adversarial` | 5     | Tests abstention on unrelated queries      |
| `ambiguous`                 | 3     | Tests abstention on underspecified input   |

The adversarial set spans three difficulty levels: orthogonal domain (FIFA World Cup),
adjacent-but-different-corpus (OSHA lockout/tagout), and plausible-but-wrong-corpus
(AISC 360 for steel stress limits). The last is the hardest and most dangerous failure
mode — the vocabulary matches closely enough that both retrieval legs score it highly.
