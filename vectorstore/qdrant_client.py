"""Thin wrapper around Qdrant. Payload doubles as the metadata store
(no separate graph/relational DB) and supports filtered re-indexing /
deletion by document_id, so re-ingesting one changed file doesn't
require a full rebuild."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from config.settings import get_settings
from core.exceptions import VectorStoreError
from core.models import Chunk, ScoredChunk
from utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.collection = settings.QDRANT_COLLECTION
        self.on_disk = settings.QDRANT_ON_DISK
        
        # Force prefer_grpc to False so it reliably communicates over HTTP (port 6333)
        grpc_port = getattr(settings, "QDRANT_GRPC_PORT", 6334)
        prefer_grpc = getattr(settings, "QDRANT_PREFER_GRPC", False)
        
        # Get timeout with retry
        timeout = getattr(settings, "QDRANT_TIMEOUT", 300)
        
        # Connect with retry logic
        self.client = self._connect_with_retry(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            timeout=timeout,
        )
        
        logger.info(
            "Connecting to Qdrant at %s:%d (gRPC=%s, timeout=%ds)",
            settings.QDRANT_HOST,
            settings.QDRANT_PORT,
            prefer_grpc,
            timeout,
        )

    def _connect_with_retry(
        self,
        host: str,
        port: int,
        grpc_port: int,
        prefer_grpc: bool,
        timeout: int,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> QdrantClient:
        """Connect to Qdrant with exponential backoff retry logic."""
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                client = QdrantClient(
                    host=host,
                    port=port,
                    grpc_port=grpc_port,
                    prefer_grpc=prefer_grpc,
                    timeout=timeout,
                )
                # Test the connection
                client.get_collections()
                
                if attempt > 0:
                    logger.info(f"✅ Qdrant connection succeeded after {attempt + 1} attempts")
                
                return client
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"⚠️ Qdrant connection attempt {attempt + 1}/{max_retries} failed: {e}"
                    )
                    logger.info(f"⏳ Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ All {max_retries} Qdrant connection attempts failed")
        
        raise VectorStoreError(
            f"Failed to connect to Qdrant after {max_retries} attempts: {last_exception}"
        )

    def collection_exists(self) -> bool:
        try:
            return self.client.collection_exists(self.collection)
        except Exception:
            return False

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        exists = self.collection_exists()
        
        if recreate and exists:
            self.client.delete_collection(self.collection)
            exists = False
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(
                    size=vector_size,
                    distance=qm.Distance.COSINE,
                    on_disk=self.on_disk,
                ),
                on_disk_payload=True,  # Keeps payload on disk to minimize RAM usage
            )
        
        logger.info(
            "Qdrant collection '%s' ready (size=%d, on_disk=%s)",
            self.collection,
            vector_size,
            self.on_disk,
        )

    def disable_indexing(self) -> None:
        """Temporarily turns off HNSW graph building during mass ingestion."""
        try:
            self.client.update_collection(
                collection_name=self.collection,
                optimizer_config=qm.OptimizersConfigDiff(indexing_threshold=0),
            )
            logger.info("Disabled Qdrant HNSW indexing threshold for fast ingestion.")
        except Exception as exc:
            logger.warning("Could not disable indexing threshold: %s", exc)

    def enable_indexing(self) -> None:
        """Re-enables HNSW index building post-ingestion."""
        try:
            self.client.update_collection(
                collection_name=self.collection,
                optimizer_config=qm.OptimizersConfigDiff(indexing_threshold=20000),
            )
            logger.info("Re-enabled Qdrant HNSW indexing threshold.")
        except Exception as exc:
            logger.warning("Could not re-enable indexing threshold: %s", exc)

    def upsert(self, chunks: List[Chunk], vectors: np.ndarray, batch_size: int = 100) -> None:
        """Insert or update points in batches to avoid timeouts and high memory usage."""
        total = len(chunks)
        for i in range(0, total, batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_vectors = vectors[i:i+batch_size]
            points = []
            for chunk, vec in zip(batch_chunks, batch_vectors):
                # chunk_id is a hex string; convert to int for Qdrant ID
                chunk_id_int = int(chunk.chunk_id, 16) % (2 ** 63) if chunk.chunk_id else None
                points.append(
                    qm.PointStruct(
                        id=chunk_id_int,
                        vector=vec.tolist() if hasattr(vec, "tolist") else list(vec),
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.document_id,
                            "text": chunk.text,
                            "content_type": chunk.content_type.value if hasattr(chunk.content_type, "value") else chunk.content_type,
                            "metadata": chunk.metadata.model_dump() if hasattr(chunk.metadata, "model_dump") else chunk.metadata,
                        },
                    )
                )
            
            # Retry upsert with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.client.upsert(collection_name=self.collection, points=points)
                    logger.info("Upserted %d/%d points", min(i + batch_size, total), total)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"⚠️ Upsert attempt {attempt + 1} failed: {e}")
                        logger.info(f"⏳ Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"❌ Upsert failed after {max_retries} attempts")
                        raise

    def delete_by_document_id(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))])
            ),
        )

    def search(self, query_vector, top_k: int) -> List[ScoredChunk]:
        from core.models import Chunk as ChunkModel

        # Retry search with exponential backoff
        max_retries = 3
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                results = self.client.search(
                    collection_name=self.collection,
                    query_vector=query_vector,
                    limit=top_k,
                )
                
                scored: List[ScoredChunk] = []
                
                for r in results:
                    payload = r.payload or {}
                    
                    # Handle content_type safely
                    content_type = payload.get("content_type", "text")
                    if hasattr(content_type, "value"):
                        content_type = content_type.value
                    
                    chunk = ChunkModel(
                        chunk_id=payload.get("chunk_id", ""),
                        document_id=payload.get("document_id", ""),
                        text=payload.get("text", ""),
                        content_type=content_type,
                        token_count=0,
                        metadata=payload.get("metadata", {}),
                    )
                    
                    scored.append(
                        ScoredChunk(
                            chunk=chunk,
                            dense_score=float(r.score),
                        )
                    )
                
                return scored
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"⚠️ Search attempt {attempt + 1} failed: {e}")
                    logger.info(f"⏳ Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Search failed after {max_retries} attempts")
        
        raise VectorStoreError(f"Search failed after {max_retries} attempts: {last_exception}")

    def stats(self) -> dict:
        try:
            info = self.client.get_collection(self.collection)
            return {"points_count": info.points_count, "collection": self.collection}
        except Exception as exc:
            return {"error": str(exc)}

    def health(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()