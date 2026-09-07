"""Reciprocal Rank Fusion. Dense (cosine) and sparse (BM25) scores live
on incomparable scales, so we fuse by rank rather than trying to
calibrate the two score distributions against each other."""
from __future__ import annotations

from typing import Dict, List

from core.models import ScoredChunk


def rrf_fuse(dense: List[ScoredChunk], sparse: List[ScoredChunk], k: int, top_n: int) -> List[ScoredChunk]:
    fused: Dict[str, ScoredChunk] = {}
    scores: Dict[str, float] = {}

    for rank, sc in enumerate(dense, start=1):
        cid = sc.chunk.chunk_id
        fused[cid] = sc
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    for rank, sc in enumerate(sparse, start=1):
        cid = sc.chunk.chunk_id
        if cid in fused:
            fused[cid].sparse_score = sc.sparse_score
        else:
            fused[cid] = sc
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    for cid, sc in fused.items():
        sc.fused_score = scores[cid]

    ranked = sorted(fused.values(), key=lambda sc: sc.fused_score or 0.0, reverse=True)
    return ranked[:top_n]
