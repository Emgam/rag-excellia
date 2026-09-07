"""
Confidence gating for hybrid RAG retrieval.
Decides whether retrieved context is strong and consistent enough to answer from,
or whether the system should refuse generation to prevent hallucination.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from config.settings import get_settings
from core.models import ScoredChunk


def _normalize_score(result: ScoredChunk) -> float:
    """
    Convert raw chunk scores to a normalized [0.0, 1.0] confidence score.

    Cross-encoder reranker scores are logits and are converted via sigmoid.
    Fused retrieval scores (RRF/Dense) are bounded to [0.0, 1.0].
    """
    # Explicitly pull rerank_score if present, fallback to final_score
    raw_score = (
        result.rerank_score
        if result.rerank_score is not None
        else result.final_score
    )

    if result.rerank_score is not None:
        # Protect against float overflow in math.exp()
        clamped_score = max(-50.0, min(50.0, raw_score))
        return 1.0 / (1.0 + math.exp(-clamped_score))

    return min(1.0, max(0.0, raw_score))


def estimate_confidence(
    results: List[ScoredChunk],
) -> Tuple[float, bool]:
    """
    Estimates retrieval confidence across top retrieved chunks using multi-chunk consensus.

    Returns:
        Tuple[float, bool]: (Calculated confidence score [0..1], Is Reliable flag)
    """
    settings = get_settings()

    # Minimum chunk threshold required for compliance (defaults to 2)
    min_required_chunks = getattr(settings, "MIN_REQUIRED_CHUNKS", 2)
    confidence_threshold = settings.CONFIDENCE_THRESHOLD

    # 1. Minimum Results Guard
    if not results or len(results) < min_required_chunks:
        return 0.0, False

    # 2. Normalize scores for up to the top 3 chunks
    scores = [_normalize_score(res) for res in results[:3]]

    top_score = scores[0]
    mean_top_scores = sum(scores) / len(scores)

    # 3. Weighted Confidence Score:
    # 70% weight on top chunk + 30% weight on supporting consensus mean
    confidence = (0.70 * top_score) + (0.30 * mean_top_scores)

    # 4. Strict Reliability Gate:
    # Both the weighted score AND the primary chunk score must pass the threshold
    reliable = (
        confidence >= confidence_threshold and top_score >= confidence_threshold
    )

    return round(float(confidence), 6), reliable