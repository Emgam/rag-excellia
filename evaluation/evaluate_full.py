#!/usr/bin/env python
"""
End‑to‑end evaluation using the production answer pipeline.
Compares with/without reranker.
Uses the `retrieved_chunks` field from QueryResponse to avoid duplicate retrieval.
Forces the 1.5B model for faster evaluation.
"""

import json
import time
import re
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict

# Increase Ollama timeout to prevent timeouts
os.environ["OLLAMA_TIMEOUT_SECONDS"] = "600"  # 10 minutes

from config.settings import get_settings
from generation.answer_service import answer_question   # your existing pipeline

# ROUGE score (install rouge-score first)
try:
    from rouge_score import rouge_scorer
except ImportError:
    print("⚠️ rouge-score not installed. Install with: pip install rouge-score")
    # Fallback dummy scorer (always returns 0)
    class DummyScorer:
        def score(self, ref, hyp):
            return {'rougeL': type('obj', (object,), {'fmeasure': 0.0})}
    rouge_scorer = lambda: DummyScorer()

# ------------------------------------------------------------------
# Load QA dataset
# ------------------------------------------------------------------
def load_qa_dataset(path: str = "evaluation/validation_qa.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_citations(answer: str) -> List[int]:
    return [int(m) for m in re.findall(r'\[(\d+)\]', answer)]

# ------------------------------------------------------------------
# Evaluate a single run (with/without reranker)
# ------------------------------------------------------------------
def evaluate_run(qa_data: List[Dict], reranker_enabled: bool = False) -> Dict:
    settings = get_settings()
    original_reranker = settings.RERANKER_ENABLED
    settings.RERANKER_ENABLED = reranker_enabled

    results = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Force the 1.5B model for all queries
    FORCED_MODEL = "qwen2.5:1.5b-instruct-q4_k_m"

    for idx, item in enumerate(qa_data):
        question = item["question"]
        true_answer = item["answer"]
        expected_sources = item.get("source_files", [])

        # ---- 1. Run the full pipeline with forced model ----
        start = time.time()
        response = answer_question(
            question,
            top_k=8,
            model=FORCED_MODEL,           # <-- force 1.5B
            temperature=0.0,
            max_tokens=200,
        )
        total_time = time.time() - start

        generated_answer = response.answer
        citations = response.sources          # list of Citation objects
        faithful = response.reliable          # True/False based on NLI
        confidence = response.confidence      # faithfulness score
        chunks = response.retrieved_chunks    # the final chunks used for generation

        # If for some reason chunks is None (shouldn't happen), we fallback to a separate retrieval
        if chunks is None:
            from retrieval.pipeline import retrieve
            chunks, _, _ = retrieve(question, top_k=8)

        # ---- 2. Retrieval metrics (using the chunks from response) ----
        if expected_sources:
            # Use getattr for metadata attributes
            found_sources = {getattr(c.chunk.metadata, "file_name", "") for c in chunks}
            recall_sources = len(found_sources & set(expected_sources)) / len(set(expected_sources)) if expected_sources else 0.0
            rank = next((i for i, c in enumerate(chunks, 1) if getattr(c.chunk.metadata, "file_name", "") in expected_sources), None)
            mrr = 1.0 / rank if rank else 0.0
        else:
            recall_sources = 0.0
            mrr = 0.0

        # ---- 3. Citation accuracy ----
        cited_ids = extract_citations(generated_answer)
        valid_citations = sum(1 for cid in cited_ids if 1 <= cid <= len(chunks))
        citation_accuracy = valid_citations / len(cited_ids) if cited_ids else 1.0

        # ---- 4. ROUGE-L ----
        rouge_scores = scorer.score(true_answer, generated_answer)
        rouge_l = rouge_scores['rougeL'].fmeasure

        # ---- 5. Overall correctness ----
        is_correct = (rouge_l > 0.3) and faithful and (citation_accuracy >= 0.8)

        results.append({
            "question": question,
            "rouge_l": rouge_l,
            "faithful": faithful,
            "nli_score": confidence,
            "citation_accuracy": citation_accuracy,
            "source_recall": recall_sources,
            "mrr": mrr,
            "total_time": total_time,
            "is_correct": is_correct,
        })

        print(f"{idx+1}/{len(qa_data)} | ROUGE={rouge_l:.3f} | Faithful={faithful} | Correct={is_correct}")

    df = pd.DataFrame(results)
    aggregate = {
        "reranker_enabled": reranker_enabled,
        "avg_rouge_l": df['rouge_l'].mean(),
        "faithfulness_rate": df['faithful'].mean(),
        "avg_citation_accuracy": df['citation_accuracy'].mean(),
        "avg_source_recall": df['source_recall'].mean(),
        "avg_mrr": df['mrr'].mean(),
        "accuracy": df['is_correct'].mean(),
        "avg_total_time": df['total_time'].mean(),
    }
    settings.RERANKER_ENABLED = original_reranker
    return aggregate, df

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    qa_path = Path("evaluation/validation_qa_enhanced.json")  # change this to your validation file
    if not qa_path.exists():
        print("⚠️ Validation QA dataset not found. Please generate it first.")
        return

    qa_data = load_qa_dataset(qa_path)
    # Filter to only simple questions
    
    # Limit to 10 for speed (adjust as needed)
    print(f"🧪 Evaluating on {len(qa_data)} simple QA queries.")

    print("\n🔹 Running WITHOUT reranker...")
    agg_off, df_off = evaluate_run(qa_data, reranker_enabled=False)
    print("Results (no reranker):", agg_off)

    print("\n🔹 Running WITH reranker...")
    agg_on, df_on = evaluate_run(qa_data, reranker_enabled=True)
    print("Results (with reranker):", agg_on)

    df_off.to_csv("evaluation/results_no_reranker.csv", index=False)
    df_on.to_csv("evaluation/results_with_reranker.csv", index=False)

    print("\n📊 Comparison:")
    print(f"{'Metric':<25} {'No Reranker':>15} {'With Reranker':>15} {'Delta':>10}")
    print("-" * 70)
    for key in agg_off.keys():
        if key == "reranker_enabled":
            continue
        val_off = agg_off[key]
        val_on = agg_on[key]
        delta = val_on - val_off
        print(f"{key:<25} {val_off:>15.4f} {val_on:>15.4f} {delta:>+10.4f}")

    print("\n✅ Recommendation:")
    if agg_on['accuracy'] > agg_off['accuracy'] + 0.02:
        print("  Enable reranker for higher accuracy (latency will increase).")
    else:
        print("  Reranker does not significantly improve accuracy – keep it off to save time.")

if __name__ == "__main__":
    main()