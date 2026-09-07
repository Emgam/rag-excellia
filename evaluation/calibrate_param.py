#!/usr/bin/env python
"""
Grid search over RAG hyperparameters for each query complexity class.
Evaluates on a validation subset and selects best config.
"""

import json
import time
import itertools
import pandas as pd
from pathlib import Path
from typing import Dict, List

from config.settings import get_settings
from generation.answer_service import answer_question
from evaluation.evaluate_full import load_qa_dataset, rouge_scorer

# Configuration
VALIDATION_QUESTIONS_PER_CLASS = 20
COMPLEXITY_CLASSES = ["simple", "medium", "complex"]  # map based on your router

# Parameter grid for each class (you can customise)
GRID = {
    "simple": {
        "top_k": [3, 5],
        "max_tokens": [80, 120],
        "reranker": [False],
        "model": ["qwen2.5:1.5b-instruct-q4_k_m"],
    },
    "medium": {
        "top_k": [5, 8],
        "max_tokens": [150, 200],
        "reranker": [False, True],
        "model": ["qwen2.5:3b-instruct-q4_k_m"],
    },
    "complex": {
        "top_k": [8, 10],
        "max_tokens": [200, 250],
        "reranker": [True],
        "model": ["qwen2.5:3b-instruct-q4_k_m"],
    },
}

def evaluate_config(qa_subset: List[Dict], config: Dict) -> Dict:
    """Run pipeline for each question and compute avg ROUGE-L and latency."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []
    latencies = []
    for item in qa_subset:
        start = time.time()
        resp = answer_question(item["question"], top_k=config["top_k"])
        elapsed = time.time() - start
        latencies.append(elapsed)
        # Compute ROUGE against ground truth
        score = scorer.score(item["answer"], resp.answer)
        rouge_scores.append(score['rougeL'].fmeasure)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    avg_latency = sum(latencies) / len(latencies)
    return {"avg_rouge": avg_rouge, "avg_latency": avg_latency}

def main():
    # Load QA dataset and split by complexity (you need a 'complexity' field)
    # For now, we'll use the hard QA set and assume all are 'complex' – adjust as needed.
    qa_data = load_qa_dataset("evaluation/qa_hard.json")
    # Here you could filter by complexity if you have labels.

    # For each class, we need a subset of questions that are labelled as that class.
    # You'll need to add a 'complexity' field to your QA JSON or use the router's classification.
    # Example: we'll just use the first N questions for each class.
    # You can also use the router's get_complexity_score to assign classes.

    results = []
    for cls in COMPLEXITY_CLASSES:
        print(f"\n🔍 Calibrating for {cls} queries...")
        # Get validation subset (simulate by taking first 20 from hard QA)
        # In practice, you'd have separate labelled subsets.
        qa_subset = qa_data[:VALIDATION_QUESTIONS_PER_CLASS]

        grid = GRID[cls]
        # Generate all combinations
        keys = list(grid.keys())
        values = list(grid.values())
        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))
            print(f"  Testing config: {config}")
            metrics = evaluate_config(qa_subset, config)
            # Compute a combined score (normalise both to 0-1)
            norm_rouge = metrics["avg_rouge"]  # already 0-1
            norm_latency = min(1.0, 5.0 / metrics["avg_latency"])  # assume 5s is ideal
            score = 0.7 * norm_rouge + 0.3 * norm_latency
            results.append({
                "class": cls,
                **config,
                "avg_rouge": metrics["avg_rouge"],
                "avg_latency": metrics["avg_latency"],
                "score": score,
            })

    # Find best per class
    df = pd.DataFrame(results)
    for cls in COMPLEXITY_CLASSES:
        best = df[df["class"] == cls].loc[df["score"].idxmax()]
        print(f"\n🏆 Best for {cls}:")
        print(best)

if __name__ == "__main__":
    main()