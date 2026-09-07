#!/usr/bin/env python
"""
Production-Ready Evaluation Script for Excellia RAG.
Measures accuracy, hallucination (refusal safety), speed (P95), and retrieval quality.
"""

import json
import time
import re
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

os.environ["OLLAMA_TIMEOUT_SECONDS"] = "600"

from config.settings import get_settings
from generation.answer_service import answer_question

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("⚠️ rouge-score not installed. Install with: pip install rouge-score")
    exit(1)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
QA_PATH = "evaluation/validation_qa_enhanced.json"
ROUGE_THRESHOLD = 0.20
LIMIT_QUESTIONS = 0    # 0 for all questions, set to 15 for a quick test
OUTPUT_CSV = "evaluation/production_results.csv"

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def load_qa_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_citations(answer: str) -> List[int]:
    return [int(m) for m in re.findall(r'\[(\d+)\]', answer)]

def get_source_files(chunks) -> set:
    """Extract unique file names from retrieved chunks."""
    if not chunks:
        return set()
    return {getattr(c.chunk.metadata, "file_name", "") for c in chunks}

# ------------------------------------------------------------------
# Main Evaluation
# ------------------------------------------------------------------
def main():
    settings = get_settings()
    qa_path = Path(QA_PATH)
    if not qa_path.exists():
        print(f"⚠️ Validation QA dataset not found: {QA_PATH}")
        return

    qa_data = load_qa_dataset(qa_path)
    if LIMIT_QUESTIONS > 0:
        qa_data = qa_data[:LIMIT_QUESTIONS]
    print(f"🧪 Evaluating on {len(qa_data)} queries (Production Mode)...")

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    results = []
    stats_by_class = defaultdict(lambda: {"total": 0, "correct": 0, "rouge_sum": 0.0, "time_sum": 0.0})

    for idx, item in enumerate(qa_data):
        question = item["question"]
        true_answer = item["answer"]
        expected_sources = set(item.get("source_files", []))
        complexity = item.get("class", "unknown")

        start = time.time()
        response = answer_question(question)
        total_time = time.time() - start

        generated_answer = response.answer
        faithful = response.reliable
        confidence = response.confidence
        chunks = response.retrieved_chunks or []

        # 1. Citation accuracy
        cited_ids = extract_citations(generated_answer)
        valid_citations = sum(1 for cid in cited_ids if 1 <= cid <= len(chunks))
        citation_accuracy = valid_citations / len(cited_ids) if cited_ids else 1.0

        # 2. Source recall (if expected sources provided)
        if expected_sources:
            found_sources = get_source_files(chunks)
            source_recall = len(found_sources & expected_sources) / len(expected_sources)
        else:
            source_recall = 0.0

        # 3. PRODUCTION LOGIC: Handle Unanswerable Questions (Hallucination Test)
        if complexity == "unanswerable":
            # It is correct ONLY if the system refused to answer
            is_correct = (settings.REFUSAL_MESSAGE in generated_answer) and faithful
            rouge_l = 1.0 if is_correct else 0.0 # Fake ROUGE score for metrics sake
        else:
            # Standard Answerable Question Logic
            rouge_scores = scorer.score(true_answer, generated_answer)
            rouge_l = rouge_scores['rougeL'].fmeasure
            is_correct = (rouge_l > ROUGE_THRESHOLD) and faithful and (citation_accuracy >= 0.8)

        results.append({
            "question": question[:60] + "...",
            "complexity": complexity,
            "rouge_l": rouge_l,
            "faithful": faithful,
            "confidence": confidence,
            "citation_acc": citation_accuracy,
            "source_recall": source_recall,
            "time": total_time,
            "correct": is_correct,
            "generated_answer": generated_answer[:150] # Save a snippet for manual review
        })

        # Update stats by class
        stats_by_class[complexity]["total"] += 1
        if is_correct:
            stats_by_class[complexity]["correct"] += 1
        stats_by_class[complexity]["rouge_sum"] += rouge_l
        stats_by_class[complexity]["time_sum"] += total_time

        status = "✅" if is_correct else "❌"
        print(f"{idx+1}/{len(qa_data)} {status} | ROUGE={rouge_l:.2f} | Time={total_time:.1f}s | Class={complexity}")

    # 4. Calculate final metrics
    df = pd.DataFrame(results)
    overall_accuracy = df['correct'].mean() * 100
    overall_rouge = df[df['complexity'] != 'unanswerable']['rouge_l'].mean() # Exclude refusals from ROUGE avg
    avg_time = df['time'].mean()
    p95_time = np.percentile(df['time'], 95) # 95th percentile latency
    faithfulness_rate = df['faithful'].mean() * 100
    avg_citation_acc = df['citation_acc'].mean()
    avg_source_recall = df['source_recall'].mean()

    print("\n" + "=" * 60)
    print("📊 PRODUCTION EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Overall Accuracy (incl. Refusals):         {overall_accuracy:.1f}%")
    print(f"  Faithfulness Rate (No Hallucinations):      {faithfulness_rate:.1f}%")
    print(f"  Average ROUGE-L Score (Answerable):        {overall_rouge:.3f}")
    print(f"  Average Citation Accuracy:                 {avg_citation_acc:.3f}")
    print(f"  Average Source Recall:                     {avg_source_recall:.3f}")
    print(f"  Average Response Time:                     {avg_time:.2f}s")
    print(f"  ⚡ P95 Response Time (95% of queries):       {p95_time:.2f}s")
    print("=" * 60)

    # 5. Breakdown by complexity
    print("\n📊 BREAKDOWN BY COMPLEXITY")
    print("-" * 60)
    for cls, stats in stats_by_class.items():
        if stats["total"] == 0:
            continue
        acc = (stats["correct"] / stats["total"]) * 100
        avg_rouge = stats["rouge_sum"] / stats["total"]
        avg_time_cls = stats["time_sum"] / stats["total"]
        print(f"  {cls.upper():<15} | Accuracy: {acc:>5.1f}% | ROUGE: {avg_rouge:.3f} | Avg Time: {avg_time_cls:.1f}s")

    # 6. Save detailed results
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Detailed results saved to {OUTPUT_CSV}")

    # 7. Summary
    if overall_accuracy >= 75 and p95_time <= 30.0:
        print("\n🎉 TARGET ACHIEVED: Accuracy ≥ 75% & P95 < 30s – Production-ready!")
    else:
        print("\n⚠️ Target not reached. Needs optimization.")

if __name__ == "__main__":
    main()