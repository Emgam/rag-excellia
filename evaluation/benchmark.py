"""
CLI benchmark runner.

Usage:
    python -m evaluation.benchmark --dataset evaluation/golden_dataset.example.json --out evaluation/report.json

Golden dataset format (list of objects):
    {"question": "...", "relevant_chunk_ids": ["<chunk_id>", ...]}

Build your own: run /ingest once, inspect chunk IDs via /stats or Qdrant
directly, and fill in relevant_chunk_ids per question in a copy of
evaluation/golden_dataset.example.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from evaluation.metrics import (
    answer_relevance, citation_accuracy, faithfulness, hallucination_rate,
    mrr, ndcg_at_k, precision_at_k, recall_at_k,
)
from generation.answer_service import answer_question
from generation.prompts import build_context
from retrieval.pipeline import retrieve


def run_benchmark(dataset_path: str, k: int = 5) -> dict:
    items = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    retrieval_scores = {"precision": [], "recall": [], "mrr": [], "ndcg": []}
    generation_scores = {"faithfulness": [], "citation_accuracy": [], "hallucination_rate": [], "answer_relevance": []}
    per_question: List[dict] = []

    for item in items:
        question = item["question"]
        relevant = set(item.get("relevant_chunk_ids", []))

        results, confidence, reliable = retrieve(question, top_k=k)
        retrieved_ids = [sc.chunk.chunk_id for sc in results]

        p = precision_at_k(retrieved_ids, relevant, k)
        r = recall_at_k(retrieved_ids, relevant, k)
        m = mrr(retrieved_ids, relevant)
        n = ndcg_at_k(retrieved_ids, relevant, k)
        for key, val in zip(retrieval_scores, (p, r, m, n)):
            retrieval_scores[key].append(val)

        response = answer_question(question, top_k=k)
        context = build_context(results)
        f = faithfulness(response.answer, context)
        c = citation_accuracy(response.answer, len(results))
        h = hallucination_rate(response.answer, context)
        ar = answer_relevance(response.answer, question)
        for key, val in zip(generation_scores, (f, c, h, ar)):
            generation_scores[key].append(val)

        per_question.append({
            "question": question, "confidence": confidence, "reliable": reliable,
            "retrieval": {"precision": p, "recall": r, "mrr": m, "ndcg": n},
            "generation": {"faithfulness": f, "citation_accuracy": c, "hallucination_rate": h, "answer_relevance": ar},
        })

    def avg(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    report = {
        "num_questions": len(items),
        "retrieval": {k_: avg(v) for k_, v in retrieval_scores.items()},
        "generation": {k_: avg(v) for k_, v in generation_scores.items()},
        "per_question": per_question,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="evaluation/report.json")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    report = run_benchmark(args.dataset, k=args.k)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written to {args.out}")
    print(json.dumps({"retrieval": report["retrieval"], "generation": report["generation"]}, indent=2))


if __name__ == "__main__":
    main()
