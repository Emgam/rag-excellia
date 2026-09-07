"""
Cross-encoder reranking using FastEmbed (ONNX). No PyTorch required.
Thread-safe singleton with support for multiple document types.
"""

from typing import List, Dict, Any, Optional, Union, TypeVar
import logging
import threading
import math
from config.settings import get_settings
from fastembed.rerank.cross_encoder import TextCrossEncoder

logger = logging.getLogger(__name__)

_RERANKER_INSTANCE: Optional[TextCrossEncoder] = None
_RERANKER_LOCK = threading.Lock()

T = TypeVar("T")


def _normalize_score(score: float) -> float:
    """
    Convert raw cross-encoder logits to [0.0, 1.0] probability.
    If the score is already between 0 and 1, return it as-is.
    """
    if score is None:
        return 0.0
    # Check if it looks like a raw logit
    if score > 1.0 or score < 0.0:
        try:
            # Apply sigmoid function
            return 1.0 / (1.0 + math.exp(-score))
        except OverflowError:
            # Handle extreme values
            return 1.0 if score > 0 else 0.0
    return float(score)


def load_reranker(force_reload: bool = False) -> Optional[TextCrossEncoder]:
    """
    Pre-loads and caches the FastEmbed TextCrossEncoder model in memory.
    Thread-safe singleton loader.
    """
    global _RERANKER_INSTANCE

    if _RERANKER_INSTANCE is not None and not force_reload:
        return _RERANKER_INSTANCE

    with _RERANKER_LOCK:
        # Double-check inside lock to prevent redundant loading
        if _RERANKER_INSTANCE is not None and not force_reload:
            return _RERANKER_INSTANCE

        settings = get_settings()

        if not getattr(settings, "RERANKER_ENABLED", True):
            logger.info("Reranker is disabled in config.")
            return None

        # MUST use the Xenova ONNX version for FastEmbed
        model_name = getattr(settings, "RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
        logger.info(f"Initializing FastEmbed reranker '{model_name}'...")

        try:
            _RERANKER_INSTANCE = TextCrossEncoder(model_name=model_name)
            logger.info(f"✅ FastEmbed Reranker '{model_name}' loaded successfully.")
        except Exception as exc:
            logger.error(f"Failed to load FastEmbed reranker '{model_name}': {exc}")
            _RERANKER_INSTANCE = None

        return _RERANKER_INSTANCE


def get_reranker() -> Optional[TextCrossEncoder]:
    """Retrieves the pre-loaded singleton model (or initializes it if missing)."""
    if _RERANKER_INSTANCE is None:
        return load_reranker()
    return _RERANKER_INSTANCE


def _extract_text(doc: Any) -> str:
    """
    Helper to extract textual content from strings, dicts, or object instances.
    Handles ScoredChunk objects properly.
    """
    if isinstance(doc, str):
        return doc

    if isinstance(doc, dict):
        return doc.get("content") or doc.get("text") or doc.get("page_content") or ""

    # Handle ScoredChunk objects - access chunk.text
    if hasattr(doc, "chunk") and hasattr(doc.chunk, "text"):
        return doc.chunk.text

    # Direct attribute lookup on object (e.g. LangChain Document)
    for attr in ["content", "text", "page_content"]:
        val = getattr(doc, attr, None)
        if isinstance(val, str):
            return val

    # Nested .chunk attribute lookup (dict version)
    if hasattr(doc, "chunk"):
        chunk = getattr(doc, "chunk")
        if isinstance(chunk, dict):
            return chunk.get("content") or chunk.get("text") or ""
        for attr in ["content", "text", "page_content"]:
            val = getattr(chunk, attr, None)
            if isinstance(val, str):
                return val

    return str(doc)


def _set_score(doc: Any, score: float) -> None:
    """
    Assigns updated normalized rerank score to dicts or objects safely.
    """
    # Score is already normalized before being passed here
    if isinstance(doc, dict):
        doc["rerank_score"] = score
        doc["score"] = score
    else:
        try:
            if hasattr(doc, "rerank_score"):
                setattr(doc, "rerank_score", score)
            if hasattr(doc, "score"):
                setattr(doc, "score", score)
            # Keep fused_score updated for legacy compatibility
            if hasattr(doc, "fused_score"):
                setattr(doc, "fused_score", score)
        except (AttributeError, TypeError):
            # Handles immutable/frozen objects (e.g. frozen dataclasses or Pydantic models)
            pass


def _get_current_score(doc: Any) -> float:
    """Gets the current score from a document for sorting fallback."""
    if isinstance(doc, dict):
        return doc.get("rerank_score", doc.get("fused_score", doc.get("score", 0.0)))
    
    if hasattr(doc, "rerank_score"):
        return getattr(doc, "rerank_score", 0.0)
    
    if hasattr(doc, "fused_score"):
        return getattr(doc, "fused_score", 0.0)
    
    if hasattr(doc, "score"):
        return getattr(doc, "score", 0.0)
    
    return 0.0


def rerank_documents(
    query: str, 
    documents: List[T], 
    top_k: Optional[int] = None
) -> List[T]:
    """
    Reranks documents in memory using FastEmbed TextCrossEncoder.
    Normalizes scores to [0.0, 1.0] probability before returning.
    """
    if not documents:
        return []

    settings = get_settings()
    if top_k is None:
        top_k = getattr(settings, "RERANK_TOP_K", 3)

    reranker = get_reranker()
    if reranker is None:
        logger.info("Reranker not available, returning top_k documents by existing score.")
        sorted_docs = sorted(documents, key=lambda d: _get_current_score(d), reverse=True)
        return sorted_docs[:top_k]

    # Extract text from documents
    doc_texts = [_extract_text(doc) for doc in documents]
    
    # Filter out empty texts
    valid_docs = []
    valid_texts = []
    for doc, text in zip(documents, doc_texts):
        if text and text.strip():
            valid_docs.append(doc)
            valid_texts.append(text)
        else:
            logger.warning(f"Skipping document with empty text during reranking: {doc}")
    
    if not valid_docs:
        logger.warning("No valid documents with text to rerank")
        return documents[:top_k]

    try:
        # FastEmbed's rerank method takes the query and a list of documents
        raw_scores = list(reranker.rerank(query, valid_texts))
        
        # Pair documents with normalized scores and update score fields
        scored_docs = []
        for doc, raw_score in zip(valid_docs, raw_scores):
            # Convert logit to [0, 1] probability
            norm_score = _normalize_score(float(raw_score))
            
            _set_score(doc, norm_score)
            scored_docs.append((doc, norm_score))

        # Sort descending by normalized score
        scored_docs.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]

    except Exception as exc:
        logger.error(f"Error during FastEmbed reranking: {exc}")
        # Fallback: sort by existing score
        sorted_docs = sorted(documents, key=lambda d: _get_current_score(d), reverse=True)
        return sorted_docs[:top_k]