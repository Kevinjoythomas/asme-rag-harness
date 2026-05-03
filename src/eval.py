"""Metrics computation and report generation.

Metrics:
- Recall@k (k=1,3,5,10) for in-corpus queries
- MRR across in-corpus queries
- Abstention precision/recall on out-of-corpus queries
- Citation graph hit rate for cross-reference queries
- Per-category breakdown
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import yaml

from src.retrieve import retrieve
from src.schema import EvalQuery, EvalReport, EvalResult, RetrievalResult

_ROOT = Path(__file__).parent.parent
QUERIES_PATH = _ROOT / "eval/queries.yaml"
REPORTS_DIR = _ROOT / "eval/reports"


def _matches(retrieved_clause: str, expected: str) -> bool:
    """True if retrieved_clause satisfies expected.

    "5" matches "5", "5.1", "5.1.2", "5#part0" — i.e. any sub-clause or split
    of the expected clause.  Does NOT match "50.1".
    """
    return (
        retrieved_clause == expected
        or retrieved_clause.startswith(expected + ".")
        or retrieved_clause.startswith(expected + "#")
    )


def _recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 1.0
    top_k = retrieved[:k]
    return 1.0 if any(_matches(r, exp) for r in top_k for exp in expected) else 0.0


def _mrr(retrieved: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    for rank, r in enumerate(retrieved, start=1):
        if any(_matches(r, exp) for exp in expected):
            return 1.0 / rank
    return 0.0


async def evaluate_query(
    eq: EvalQuery,
    threshold: float = 0.020,
    k: int = 10,
) -> EvalResult:
    result: RetrievalResult = await retrieve(
        eq.query,
        k=k,
        confidence_threshold=threshold,
        expand_references=True,
    )

    retrieved_clauses = [c.clause for c in result.chunks]

    r1 = _recall_at_k(retrieved_clauses, eq.expected_clauses, 1)
    r3 = _recall_at_k(retrieved_clauses, eq.expected_clauses, 3)
    r5 = _recall_at_k(retrieved_clauses, eq.expected_clauses, 5)
    r10 = _recall_at_k(retrieved_clauses, eq.expected_clauses, 10)
    rr = _mrr(retrieved_clauses, eq.expected_clauses)

    citation_hit: bool | None = None
    if eq.expected_referenced_clauses:
        citation_hit = any(
            any(r == ref or r.startswith(ref + "#") for r in retrieved_clauses)
            for ref in eq.expected_referenced_clauses
        )

    if eq.expect_abstain:
        correct = result.abstain
    else:
        correct = r5 > 0.0

    return EvalResult(
        query_id=eq.id,
        category=eq.category,
        query=eq.query,
        retrieved_clause_ids=retrieved_clauses,
        expected_clauses=eq.expected_clauses,
        abstained=result.abstain,
        expected_abstain=eq.expect_abstain,
        recall_at_1=r1,
        recall_at_3=r3,
        recall_at_5=r5,
        recall_at_10=r10,
        reciprocal_rank=rr,
        citation_hit=citation_hit,
        correct=correct,
    )


def compute_report(results: list[EvalResult]) -> EvalReport:
    in_corpus = [r for r in results if not r.expected_abstain]
    should_abstain = [r for r in results if r.expected_abstain]
    cross_ref = [r for r in results if r.citation_hit is not None]

    def mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    actually_abstained = [r for r in results if r.abstained]
    abs_precision = (
        sum(1 for r in actually_abstained if r.expected_abstain) / len(actually_abstained)
        if actually_abstained else 0.0
    )
    abs_recall = (
        sum(1 for r in should_abstain if r.abstained) / len(should_abstain)
        if should_abstain else 0.0
    )
    citation_rate = (
        sum(1 for r in cross_ref if r.citation_hit is True) / len(cross_ref)
        if cross_ref else 0.0
    )

    categories = [
        "in_corpus_direct",
        "in_corpus_cross_reference",
        "out_of_corpus_adversarial",
        "ambiguous",
    ]
    per_cat: dict[str, dict[str, float]] = {}
    for cat in categories:
        cat_r = [r for r in results if r.category == cat]
        if not cat_r:
            continue
        per_cat[cat] = {
            "recall_at_5": mean([r.recall_at_5 for r in cat_r]),
            "mrr": mean([r.reciprocal_rank for r in cat_r]),
            "correct_rate": mean([float(r.correct) for r in cat_r]),
            "n": float(len(cat_r)),
        }

    return EvalReport(
        timestamp=datetime.utcnow().isoformat(),
        total_queries=len(results),
        in_corpus_recall_at_1=mean([r.recall_at_1 for r in in_corpus]),
        in_corpus_recall_at_3=mean([r.recall_at_3 for r in in_corpus]),
        in_corpus_recall_at_5=mean([r.recall_at_5 for r in in_corpus]),
        in_corpus_recall_at_10=mean([r.recall_at_10 for r in in_corpus]),
        in_corpus_mrr=mean([r.reciprocal_rank for r in in_corpus]),
        abstention_precision=abs_precision,
        abstention_recall=abs_recall,
        citation_traversal_hit_rate=citation_rate,
        per_category=per_cat,
        results=results,
    )


def _format_md(report: EvalReport) -> str:
    lines = [
        f"# Eval Report — {report.timestamp}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| In-corpus Recall@1 | {report.in_corpus_recall_at_1:.3f} |",
        f"| In-corpus Recall@3 | {report.in_corpus_recall_at_3:.3f} |",
        f"| In-corpus Recall@5 | {report.in_corpus_recall_at_5:.3f} |",
        f"| In-corpus Recall@10 | {report.in_corpus_recall_at_10:.3f} |",
        f"| In-corpus MRR | {report.in_corpus_mrr:.3f} |",
        f"| Abstention Precision | {report.abstention_precision:.3f} |",
        f"| Abstention Recall | {report.abstention_recall:.3f} |",
        f"| Citation Traversal Hit Rate | {report.citation_traversal_hit_rate:.3f} |",
        "",
        "## Per-Category Breakdown",
        "",
        "| Category | Recall@5 | MRR | Correct Rate | N |",
        "|----------|----------|-----|--------------|---|",
    ]
    for cat, m in report.per_category.items():
        lines.append(
            f"| {cat} | {m['recall_at_5']:.3f} | {m['mrr']:.3f} | {m['correct_rate']:.3f} | {int(m['n'])} |"
        )
    lines += [
        "",
        "## Per-Query Results",
        "",
        "| ID | Category | Abstain | Expected | R@5 | MRR | Citation |",
        "|----|----------|---------|----------|-----|-----|---------|",
    ]
    for r in report.results:
        cit = "-" if r.citation_hit is None else ("✓" if r.citation_hit else "✗")
        lines.append(
            f"| {r.query_id} | {r.category[:20]} "
            f"| {'✓' if r.abstained else '✗'} "
            f"| {'✓' if r.expected_abstain else '✗'} "
            f"| {r.recall_at_5:.2f} | {r.reciprocal_rank:.2f} | {cit} |"
        )
    return "\n".join(lines)


async def main(
    queries_path: Path = QUERIES_PATH,
    threshold: float = 0.40,
    k: int = 10,
    debug: bool = False,
) -> EvalReport:
    queries = [EvalQuery(**q) for q in yaml.safe_load(queries_path.read_text(encoding="utf-8"))]
    print(f"Evaluating {len(queries)} queries at threshold={threshold}...")

    results: list[EvalResult] = []
    for eq in queries:
        res = await evaluate_query(eq, threshold=threshold, k=k)
        tag = "ABSTAIN" if res.abstained else f"R@5={res.recall_at_5:.2f}"
        print(f"  {eq.id} [{eq.category[:18]}]  {tag}")
        if debug and not res.abstained and res.recall_at_5 < 1.0 and eq.expected_clauses:
            print(f"    expected : {eq.expected_clauses}")
            print(f"    retrieved: {res.retrieved_clause_ids[:5]}")
        results.append(res)

    report = compute_report(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    (REPORTS_DIR / f"eval_{ts}.md").write_text(_format_md(report), encoding="utf-8")
    (REPORTS_DIR / f"eval_{ts}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nIn-corpus Recall@5: {report.in_corpus_recall_at_5:.3f}")
    print(f"In-corpus MRR:      {report.in_corpus_mrr:.3f}")
    print(f"Abstention Recall:  {report.abstention_recall:.3f}")
    print(f"Report -> eval/reports/eval_{ts}.md")
    return report


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    args = sys.argv[1:]
    debug = "--debug" in args
    args = [a for a in args if a != "--debug"]
    t = float(args[0]) if args else 0.40
    asyncio.run(main(threshold=t, debug=debug))
