"""Lightweight confidence estimation for hybrid RAG retrieval."""

from __future__ import annotations

import math
from typing import List, Tuple

from config.settings import get_settings
from core.models import ScoredChunk


def _normalize_score(result: ScoredChunk) -> float:
    """
    Convert the final score to the range [0, 1].

    Reranker scores are logits and are converted with a sigmoid.
    Fused retrieval scores are already treated as confidence-like values.
    """
    score = result.final_score

    if result.rerank_score is not None:
        # Protect against extreme values.
        score = max(-50.0, min(50.0, score))
        return 1.0 / (1.0 + math.exp(-score))

    return min(1.0, max(0.0, score))


def estimate_confidence(
    results: List[ScoredChunk],
) -> Tuple[float, bool]:
    settings = get_settings()

    if not results:
        return 0.0, False

    scores = [
        _normalize_score(result)
        for result in results[:3]
    ]

    top_score = scores[0]
    mean_top_scores = sum(scores) / len(scores)

    # The top result is more important, but support from other
    # retrieved chunks also contributes to confidence.
    confidence = (
        0.70 * top_score
        + 0.30 * mean_top_scores
    )

    reliable = (
        confidence >= settings.CONFIDENCE_THRESHOLD
    )

    return confidence, reliable