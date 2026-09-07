"""
Orchestrates the full retrieval flow for high-compliance documentation:
dynamic top-k -> dense + sparse search (parallelized) 
-> RRF fusion -> cross-encoder rerank -> strict relevance thresholding 
-> confidence gate & early refusal.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from config.settings import get_settings
from core.models import ScoredChunk
from embeddings.embedder import get_embedder
from retrieval.confidence.confidence import estimate_confidence
from retrieval.fusion.rrf import rrf_fuse
from retrieval.reranking.reranker import rerank_documents
from retrieval.sparse.bm25_index import get_bm25_index
from utils.logger import get_logger
from utils.timer import Timer, timeit
from vectorstore.qdrant_client import get_vector_store

logger = get_logger(__name__)

# Thread pool for parallel execution
_executor = ThreadPoolExecutor(max_workers=4)


def _normalize_score(score: float | None) -> float:
    """
    Convert cross-encoder raw logits to [0.0, 1.0] probability.
    If the score is already between 0 and 1, return it as-is.
    """
    if score is None:
        return 0.0
    if score > 1.0 or score < 0.0:
        # Apply sigmoid function
        return 1.0 / (1.0 + math.exp(-score))
    return float(score)


def _parallel_search(store, bm25, query_vector, question, settings):
    """
    Run dense and sparse search in parallel using ThreadPoolExecutor.
    """
    dense_future = _executor.submit(
        store.search,
        query_vector,
        settings.DENSE_TOP_K
    )
    
    sparse_future = _executor.submit(
        bm25.search,
        question,
        settings.SPARSE_TOP_K
    )
    
    dense_results = dense_future.result()
    sparse_results = sparse_future.result()
    
    return dense_results, sparse_results


@timeit
def retrieve(
    question: str,
    top_k: int | None = None,
    fused_top_n: int | None = None,   # <-- NEW PARAMETER
    reranker_enabled: bool | None = None,   # <-- OPTIONAL: for per-route control
) -> Tuple[List[ScoredChunk], float, bool]:
    """
    Full zero-hallucination retrieval pipeline with strict relevance gating.
    
    Args:
        question: User's question
        top_k: Number of chunks to return (determined by SmartRouter)
        fused_top_n: Number of fused candidates to keep (default from settings)
        reranker_enabled: Override for reranker (default from settings)
    
    Returns:
        Tuple of (list of scored chunks, confidence score, reliability flag)
    """
    settings = get_settings()
    
    # Single source of truth for compliance thresholds
    MIN_RELEVANCE_SCORE = getattr(settings, "MIN_RELEVANCE_SCORE", 0.35)
    MIN_REQUIRED_CHUNKS = getattr(settings, "MIN_REQUIRED_CHUNKS", 1)
    
    with Timer("retrieve_total"):
        
        # 1. Setup/load components
        with Timer("retrieve_setup"):
            embedder = get_embedder()
            store = get_vector_store()
            bm25 = get_bm25_index()
        
        # 2. Use top_k passed by SmartRouter
        final_top_k = top_k if top_k is not None else 5
        
        # 3. Query embedding
        with Timer("retrieve_embedding"):
            query_vector = embedder.embed_query(question)
        
        # 4. Dense + Sparse search in PARALLEL
        with Timer("retrieve_parallel_search"):
            dense_results, sparse_results = _parallel_search(
                store, bm25, query_vector, question, settings
            )
        
        # 5. RRF Fusion - use passed fused_top_n or fallback to settings
        with Timer("retrieve_rrf_fusion"):
            fused = rrf_fuse(
                dense_results,
                sparse_results,
                k=settings.RRF_K,
                top_n=fused_top_n if fused_top_n is not None else settings.FUSED_TOP_N,
            )
        
        # 6. Reranking - use passed reranker_enabled or fallback to settings
        with Timer("retrieve_reranking"):
            use_reranker = reranker_enabled if reranker_enabled is not None else settings.RERANKER_ENABLED
            if use_reranker and fused:
                try:
                    reranked = rerank_documents(
                        question,
                        fused,
                        top_k=final_top_k,
                    )
                except Exception as exc:
                    logger.error(f"Reranker failed critically: {exc}. Falling back with strict score warning.")
                    reranked = fused[:final_top_k]
            else:
                reranked = fused[:final_top_k]
        
        # 7. STRICT ANTI-HALLUCINATION GATE: Hard Score Thresholding
        with Timer("retrieve_relevance_filter"):
            valid_chunks = []
            for chunk in reranked:
                # Prioritize normalized rerank score, fallback to dense score
                score = getattr(chunk, 'rerank_score', None)
                if score is not None:
                    norm_score = _normalize_score(score)
                else:
                    norm_score = float(chunk.dense_score) if chunk.dense_score is not None else 0.0
                
                # Update the chunk's rerank_score to the normalized version for downstream use
                if hasattr(chunk, 'rerank_score'):
                    chunk.rerank_score = norm_score
                
                if norm_score >= MIN_RELEVANCE_SCORE:
                    valid_chunks.append(chunk)
        
        # 8. Confidence estimation
        with Timer("retrieve_confidence"):
            confidence, reliable = estimate_confidence(valid_chunks)
            
        # 9. EARLY REFUSAL GATE
        if len(valid_chunks) < MIN_REQUIRED_CHUNKS or not reliable:
            logger.warning(
                "Early Refusal Triggered | Query: '%s' | Valid Chunks: %d | Confidence: %.4f | Reliable: %s",
                question, len(valid_chunks), confidence, reliable
            )
            return [], confidence, False

    # Log summary
    logger.info(
        "Retrieval Success | dense=%d | sparse=%d | fused=%d | valid=%d | confidence=%.6f | reliable=%s",
        len(dense_results),
        len(sparse_results),
        len(fused),
        len(valid_chunks),
        confidence,
        reliable,
    )
    
    return valid_chunks, confidence, True