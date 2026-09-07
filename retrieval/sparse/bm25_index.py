"""BM25 sparse index (rank_bm25) — catches exact keyword/identifier/
error-code matches that dense embeddings can blur out. Persisted to
disk as a pickle so it survives API restarts without a full reindex."""
from __future__ import annotations

import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import List

from config.settings import get_settings
from core.models import Chunk, ScoredChunk
from utils.logger import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self._bm25 = None

    def build(self, chunks: List[Chunk]) -> None:
        """Build the BM25 index from a list of Chunk objects."""
        from rank_bm25 import BM25Okapi

        self.chunks = chunks if chunks is not None else []
        if not self.chunks:
            self._bm25 = None
            logger.info("BM25 built with 0 chunks (empty index)")
            return

        tokenized = [tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 built with {len(self.chunks)} chunks")

    def add(self, chunks: List[Chunk]) -> None:
        """Append chunks to the index (rebuilds from scratch)."""
        if not chunks:
            return
        combined = self.chunks + chunks
        self.build(combined)

    def remove_by_document_id(self, document_id: str) -> None:
        """Remove all chunks belonging to a document."""
        remaining = [c for c in self.chunks if c.document_id != document_id]
        self.build(remaining)

    def search(self, query: str, top_k: int) -> List[ScoredChunk]:
        """Return top_k ScoredChunks with BM25 scores."""
        if not self._bm25 or not self.chunks:
            return []

        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]

        result = []
        for c, s in ranked:
            if s > 0:
                result.append(ScoredChunk(chunk=c, sparse_score=float(s)))
        return result

    def save(self) -> None:
        """Persist only the chunk list (BM25Okapi is rebuilt on load)."""
        settings = get_settings()
        path = Path(settings.BM25_INDEX_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info("Saved BM25 index (%d chunks) to %s", len(self.chunks), path)

    def load(self) -> bool:
        """Load chunk list from disk and rebuild BM25Okapi."""
        settings = get_settings()
        path = Path(settings.BM25_INDEX_PATH)
        if not path.exists():
            logger.info("No BM25 index file found at %s", path)
            return False

        with open(path, "rb") as f:
            chunks = pickle.load(f)
        self.build(chunks)
        logger.info("Loaded BM25 index (%d chunks) from %s", len(chunks), path)
        return True


@lru_cache
def get_bm25_index() -> BM25Index:
    idx = BM25Index()
    idx.load()
    return idx