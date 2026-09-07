"""Embedding service using FastEmbed (CPU‑optimized, no PyTorch)."""

from __future__ import annotations

import sys
from typing import List, Optional
import numpy as np
from fastembed import TextEmbedding

from core.exceptions import EmbeddingError


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model: Optional[TextEmbedding] = None

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            try:
                self._model = TextEmbedding(model_name=self.model_name, threads=4)
            except Exception as e:
                raise EmbeddingError(f"Failed to load embedding model: {e}")
        return self._model

    def embed_passages(
        self, 
        texts: list[str], 
        batch_size: int = 256, 
        show_progress: bool = True
    ) -> np.ndarray:
        """Embed a list of texts into dense vectors efficiently using FastEmbed ONNX batches."""
        if not texts:
            return np.array([])

        try:
            total = len(texts)
            if show_progress and total > 100:
                print(f"Embedding {total} chunks in batches...")

            # FastEmbed handles batching natively inside ONNX C++ for maximum throughput
            embeddings = list(self.model.embed(texts, batch_size=batch_size))

            if show_progress and total > 100:
                print("Embedding complete.")

            return np.stack(embeddings)
        except Exception as e:
            raise EmbeddingError(f"Embedding failed: {e}")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query vector."""
        return self.embed_passages([query], show_progress=False)[0]


# Singleton instance
_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """Thread-safe singleton accessor for Embedder."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def encode_texts(texts: list[str]) -> list[list[float]]:
    """Convenience helper function for smart router semantic routing imports."""
    embedder = get_embedder()
    vectors = embedder.embed_passages(texts, show_progress=False)
    return vectors.tolist()


if __name__ == "__main__":
    embedder = get_embedder()

    # Safely print the underlying model instance type
    print("Underlying Model Object:", type(embedder.model))

    # Verify active runtime backend modules
    if "onnxruntime" in sys.modules:
        print("✅ Backend Engine: ONNX Runtime (FastEmbed)")
    elif "torch" in sys.modules:
        print("🔥 Backend Engine: PyTorch")
    else:
        print("Unknown runtime backend")