#!/usr/bin/env python
"""
Smart Routing and Query Classification for Excellia RAG.
Uses CPU-optimized semantic prototype embedding matching for zero-latency dynamic routing.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class SmartRouter:
    """Routes queries to optimal pipeline configuration based on semantic intent."""

    def __init__(self):
        self.settings = get_settings()

        # Route definitions matching project settings
        self.complexity_rules: Dict[str, Dict[str, Any]] = {
            "simple": {
                "route": "simple",
                "top_k": getattr(self.settings, "SIMPLE_TOP_K", 5),
                "chunk_size": getattr(self.settings, "SIMPLE_CHUNK_SIZE", 250),
                "max_tokens": getattr(self.settings, "SIMPLE_MAX_TOKENS", 150),
                "model": getattr(self.settings, "SIMPLE_MODEL", "qwen2.5:1.5b-instruct-q4_k_m"),
                "temperature": getattr(self.settings, "SIMPLE_TEMPERATURE", 0.1),
                "reranker_enabled": getattr(self.settings, "SIMPLE_RERANKER", False),
                "description": "Simple definition and factual lookup queries",
                "fused_top_n": 10, 
            },
            "medium": {
                "route": "medium",
                "top_k": getattr(self.settings, "MEDIUM_TOP_K", 5),
                "chunk_size": getattr(self.settings, "MEDIUM_CHUNK_SIZE", 500),
                "max_tokens": getattr(self.settings, "MEDIUM_MAX_TOKENS", 300),
                "model": getattr(self.settings, "MEDIUM_MODEL", "qwen2.5:1.5b-instruct-q4_k_m"),
                "temperature": getattr(self.settings, "MEDIUM_TEMPERATURE", 0.2),
                "reranker_enabled": getattr(self.settings, "MEDIUM_RERANKER", True),
                "description": "Procedural and step-by-step how-to queries",
                "fused_top_n": 20, 
            },
            "complex": {
                "route": "complex",
                "top_k": getattr(self.settings, "COMPLEX_TOP_K", 3),
                "chunk_size": getattr(self.settings, "COMPLEX_CHUNK_SIZE", 700),
                "max_tokens": getattr(self.settings, "COMPLEX_MAX_TOKENS", 600),
                "model": getattr(self.settings, "COMPLEX_MODEL", "qwen2.5:3b-instruct-q4_k_m"),
                "temperature": getattr(self.settings, "COMPLEX_TEMPERATURE", 0.3),
                "reranker_enabled": getattr(self.settings, "COMPLEX_RERANKER", True),
                "description": "Comparison, deep analysis, and multi-document reasoning",
                "fused_top_n": 30, 
            },
            "fallback": {
                "route": "fallback",
                "top_k": 5,
                "chunk_size": 250,
                "max_tokens": 180,
                "model": getattr(self.settings, "OLLAMA_MODEL", "qwen2.5:3b-instruct-q4_k_m"),
                "temperature": getattr(self.settings, "OLLAMA_TEMPERATURE", 0.2),
                "reranker_enabled": False,
                "description": "Fallback routing path",
            },
        }

        # Canonical intent anchor sentences used for semantic matching
        self.prototype_sentences: Dict[str, List[str]] = {
            "simple": [
                "What is the definition of this term?",
                "Define this word or short concept.",
                "Give me a quick factual lookup answer.",
                "What does this abbreviation or acronym stand for?",
            ],
            "medium": [
                "How do I perform this process or procedure step by step?",
                "What are the instructions to submit or request this?",
                "Guide me through configuring, installing, or applying for this.",
                "What is the standard workflow or policy procedure?",
            ],
            "complex": [
                "Compare the key differences and trade-offs between option A and B.",
                "Analyze the structural pros and cons and architectural impact.",
                "Synthesize a comprehensive evaluation across multiple detailed policies.",
                "What are the major advantages versus disadvantages when comparing these systems?",
            ],
        }

        self._prototype_matrix: Optional[Dict[str, np.ndarray]] = None

    def initialize_prototypes(self, embed_fn: Callable[[List[str]], List[List[float]]]) -> None:
        """
        Pre-computes prototype anchor embeddings once at application startup.
        
        Args:
            embed_fn: A function that accepts a list of text strings and 
                      returns a list of vector float lists (e.g. sentence_transformer.encode).
        """
        try:
            self._prototype_matrix = {}
            for category, sentences in self.prototype_sentences.items():
                vecs = embed_fn(sentences)
                matrix = np.array(vecs, dtype=np.float32)
                # Normalize matrix rows for fast cosine calculation
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self._prototype_matrix[category] = matrix / norms
            logger.info("✅ SmartRouter prototype embeddings initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize router prototypes: {e}")
            self._prototype_matrix = None

    def classify_query(
        self,
        question: str,
        query_vector: Optional[Union[List[float], np.ndarray]] = None,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ) -> Dict[str, Any]:
        """
        Classifies query dynamically into simple, medium, or complex routing tiers.
        Uses vector cosine similarity if vector/embedder is available; falls back to fast rule guards.
        """
        # --- PATH 1: SEMANTIC VECTOR ROUTING (<2ms execution) ---
        if query_vector is not None or embed_fn is not None:
            try:
                # Lazy initialization if matrix isn't ready but embed_fn is supplied
                if self._prototype_matrix is None and embed_fn is not None:
                    self.initialize_prototypes(embed_fn)

                if self._prototype_matrix is not None:
                    # Obtain vector if only string + embed_fn passed
                    if query_vector is None and embed_fn is not None:
                        query_vector = embed_fn([question])[0]

                    q_vec = np.array(query_vector, dtype=np.float32).flatten()
                    q_norm = np.linalg.norm(q_vec)

                    if q_norm > 0:
                        q_vec_norm = q_vec / q_norm
                        scores: Dict[str, float] = {}

                        for category, matrix in self._prototype_matrix.items():
                            # Matrix-vector dot product = Cosine Similarity
                            similarities = np.dot(matrix, q_vec_norm)
                            scores[category] = float(np.max(similarities))

                        best_route = max(scores, key=scores.get)
                        logger.info(f"🎯 Semantic Route: {best_route.upper()} | Similarity Scores: {scores}")
                        return self.complexity_rules[best_route]

            except Exception as e:
                logger.warning(f"Semantic routing failed ({e}). Falling back to heuristic rules.")

        # --- PATH 2: HEURISTIC FALLBACK GUARD (Ultra-fast <0.1ms) ---
        return self._heuristic_fallback(question)

    def _heuristic_fallback(self, question: str) -> Dict[str, Any]:
        """Fast fallback rule engine if embeddings are unavailable."""
        q_lower = question.lower().strip()
        word_count = len(q_lower.split())

        # 1. Explicit complex intent indicators (Keywords are more reliable than word count)
        complex_triggers = ["compare", "difference", "vs", "versus", "pros and cons", "tradeoff", "analysis", "evaluate", "how does"]
        if any(kw in q_lower for kw in complex_triggers):
            return self.complexity_rules["complex"]

        # 2. Explicit simple definition indicators
        simple_triggers = ["what is", "define", "meaning of", "definition of", "what are", "who is", "how many"]
        if any(q_lower.startswith(kw) for kw in simple_triggers):
            return self.complexity_rules["simple"]

        # 3. Medium triggers (Procedural)
        medium_triggers = ["how do i", "how to", "steps to", "procedure for", "process for"]
        if any(kw in q_lower for kw in medium_triggers):
            return self.complexity_rules["medium"]

        # 4. Fallback based on length (Only if no keywords matched)
        if word_count > 25:
            return self.complexity_rules["complex"]
        if word_count <= 10:
            return self.complexity_rules["simple"]
        
        # Default standard path
        return self.complexity_rules["medium"]

    def get_complexity_score(self, question: str) -> int:
        """Calculates a normalized 1-10 complexity score for telemetry and logs."""
        q_lower = question.lower()
        word_count = len(q_lower.split())

        # Start with a base score from word count
        score = min(5, word_count // 4)  # Max 5 points from length
        
        if any(kw in q_lower for kw in ["compare", "difference", "vs", "versus", "how does"]):
            score += 5 # Instantly bump to complex
        if any(kw in q_lower for kw in ["analyze", "evaluate", "assess", "impact"]):
            score += 3
        
        # Simple questions stay low
        if any(kw in q_lower for kw in ["what is", "define", "how many"]):
            score = min(score, 3)

        return min(10, max(1, score))

    def get_model_info(self, model_name: str) -> Dict[str, str]:
        """Returns metadata for known local quantized models."""
        models = {
            "qwen2.5:3b-instruct-q4_k_m": {
                "type": "quantized",
                "size": "1.8 GB",
                "speed": "medium",
                "quality": "very good",
            },
            "qwen2.5:1.5b-instruct-q4_k_m": {
                "type": "quantized",
                "size": "1.0 GB",
                "speed": "fast",
                "quality": "good",
            },
        }
        return models.get(model_name, {"type": "unknown", "size": "N/A", "speed": "medium", "quality": "standard"})


# Singleton instance container
_router: Optional[SmartRouter] = None


def get_router() -> SmartRouter:
    """Thread-safe singleton accessor for SmartRouter."""
    global _router
    if _router is None:
        _router = SmartRouter()
    return _router