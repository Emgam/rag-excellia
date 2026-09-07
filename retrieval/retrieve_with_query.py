"""
Enhanced retrieval with smart routing.
Wraps the existing retrieve function with dynamic parameters and semantic classification.
"""

from __future__ import annotations

import importlib
from typing import Callable, List, Optional, Tuple, Union
import numpy as np

from config.settings import get_settings
from core.models import ScoredChunk
from generation.smart_routing import get_router
from retrieval.pipeline import retrieve as base_retrieve
from utils.logger import get_logger
from utils.timer import Timer, timeit

logger = get_logger(__name__)


def _get_chunk_score(chunk: ScoredChunk) -> float:
    """
    Safely extracts the relevance score from a chunk, checking alternative score
    property names and handling None values to avoid comparison errors.
    """
    score = getattr(chunk, "score", None)
    if score is None:
        score = getattr(chunk, "rerank_score", None)
    if score is None:
        score = getattr(chunk, "dense_score", None)
    return float(score) if score is not None else 0.0


def _get_default_embed_fn() -> Optional[Callable[[List[str]], List[List[float]]]]:
    """
    Dynamically resolves and returns an embedding function across potential module paths.
    Tries importing `encode_texts` first, then falls back to `get_embedder()`.
    """
    module_paths = [
        "retrieval.embedder",
        "embeddings.embedder",
        "core.embedder",
        "retrieval.embeddings",
        "core.embeddings",
    ]

    # Priority 1: Direct `encode_texts` helper function
    for path in module_paths:
        try:
            mod = importlib.import_module(path)
            if hasattr(mod, "encode_texts"):
                logger.debug(f"Found 'encode_texts' at {path}")
                return getattr(mod, "encode_texts")
        except ImportError:
            continue

    # Priority 2: Wrap `get_embedder().embed_passages` singleton
    for path in module_paths:
        try:
            mod = importlib.import_module(path)
            if hasattr(mod, "get_embedder"):
                logger.debug(f"Found 'get_embedder' singleton at {path}")
                get_embedder_fn = getattr(mod, "get_embedder")
                return lambda texts: get_embedder_fn().embed_passages(texts, show_progress=False).tolist()
        except ImportError:
            continue

    logger.warning("⚠️ Could not auto-resolve default embedding function from project paths.")
    return None


@timeit
def retrieve_smart(
    question: str,
    top_k: Optional[int] = None,
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    query_vector: Optional[Union[List[float], np.ndarray]] = None,
) -> Tuple[List[ScoredChunk], float, bool]:
    """
    Enhanced retrieval with smart routing.
    Uses semantic embedding classification to select dynamic execution parameters.

    Args:
        question: User query string.
        top_k: Optional explicit override for top_k results.
        embed_fn: Optional embedding encoder function for semantic routing.
        query_vector: Optional precomputed query vector for zero-latency routing.
    """
    settings = get_settings()
    router = get_router()

    # Automatically resolve default embedder if neither vector nor embed_fn was provided
    if embed_fn is None and query_vector is None:
        embed_fn = _get_default_embed_fn()

    # ============================================================
    # 1. CLASSIFY QUERY WITH SEMANTIC ROUTER
    # ============================================================
    config = router.classify_query(
        question,
        query_vector=query_vector,
        embed_fn=embed_fn,
    )
    complexity = router.get_complexity_score(question)

    logger.info(f"📊 Query Complexity: {complexity}/10")
    logger.info(f"🎯 Route: {config.get('description', 'unknown')}")
    logger.info(f"🔢 Top K: {config.get('top_k', 3)}")

    # ============================================================
    # 2. UPDATE SETTINGS FOR THIS QUERY
    # ============================================================
    # Store original settings with safe fallbacks
    original_dense = getattr(settings, "DENSE_TOP_K", 30)
    original_sparse = getattr(settings, "SPARSE_TOP_K", 30)
    original_reranker = getattr(settings, "RERANKER_ENABLED", True)
    original_fused = getattr(settings, "FUSED_TOP_N", 15)

    try:
        # Apply dynamic settings from routing config
        if config.get("top_k"):
            settings.DENSE_TOP_K = config["top_k"]
            settings.SPARSE_TOP_K = config["top_k"]
            logger.info(f"📊 DENSE_TOP_K set to: {config['top_k']}")

        if config.get("reranker_enabled") is not None:
            settings.RERANKER_ENABLED = config["reranker_enabled"]
            logger.info(f"🔄 Reranker: {'ON' if config['reranker_enabled'] else 'OFF'}")

        # Route-specific parameter tuning
        route = config.get("route")
        if route == "simple":
            settings.FUSED_TOP_N = 2
            logger.info("⚡ Simple query mode - FUSED_TOP_N: 2")
        elif route == "complex":
            settings.FUSED_TOP_N = 6
            logger.info("🔬 Complex query mode - FUSED_TOP_N: 6")

        # ============================================================
        # 3. CALL BASE RETRIEVE
        # ============================================================
        target_top_k = top_k if top_k is not None else config.get("top_k", 3)

        with Timer("retrieve_with_routing"):
            chunks, confidence, reliable = base_retrieve(
                question,
                top_k=target_top_k,
            )

        # ============================================================
        # 4. FILTER LOW-CONFIDENCE CHUNKS
        # ============================================================
        min_score = 0.01  # Threshold for chunk relevance
        filtered_chunks = [c for c in chunks if _get_chunk_score(c) > min_score]

        if len(filtered_chunks) < len(chunks):
            logger.info(f"🧹 Filtered {len(chunks) - len(filtered_chunks)} low-confidence chunks")
            chunks = filtered_chunks

        return chunks, confidence, reliable

    except Exception as e:
        logger.error(f"❌ Error in smart retrieval: {e}")
        # Fallback to default base retrieval
        return base_retrieve(question, top_k=top_k or 3)

    finally:
        # ============================================================
        # 5. RESTORE ORIGINAL SETTINGS
        # ============================================================
        settings.DENSE_TOP_K = original_dense
        settings.SPARSE_TOP_K = original_sparse
        settings.RERANKER_ENABLED = original_reranker
        settings.FUSED_TOP_N = original_fused
        logger.debug("✅ Restored original settings")


def get_dynamic_top_k_with_routing(
    question: str,
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    query_vector: Optional[Union[List[float], np.ndarray]] = None,
) -> int:
    """
    Get dynamic top_k based on semantic query complexity and routing.
    """
    router = get_router()
    if embed_fn is None and query_vector is None:
        embed_fn = _get_default_embed_fn()

    config = router.classify_query(
        question,
        query_vector=query_vector,
        embed_fn=embed_fn,
    )
    return config.get("top_k", 10)