"""
Top-level ingestion orchestration used by the /ingest and /reindex
API endpoints: discover -> parse -> chunk -> embed -> index (dense + sparse),
with streaming batching to prevent RAM/OOM crashes and per-file error isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from config.settings import get_settings
from core.exceptions import ExcelliaError
from core.models import Chunk, IngestResponse
from embeddings.embedder import get_embedder
from ingestion.chunking.chunker import chunk_block
from ingestion.discovery.discover import discover_files
from ingestion.metadata.metadata import build_metadata, document_id_for
from ingestion.parsers.parser import parse_document
from retrieval.sparse.bm25_index import get_bm25_index
from utils.logger import get_logger
from vectorstore.qdrant_client import get_vector_store

logger = get_logger(__name__)

# Optimal batch size for ONNX/PyTorch embedding and Qdrant upserts
BATCH_SIZE = 32


def ingest(path: str | None = None, recreate: bool = False) -> IngestResponse:
    settings = get_settings()
    root = Path(path).resolve() if path else settings.documents_path
    logger.info(f"Ingestion starting: root={root}, exists? {root.exists()}")

    # --- Discovery ---
    files = discover_files(root)
    logger.info(f"Discovered {len(files)} files")
    if not files:
        logger.warning("No files discovered – check path and SUPPORTED_EXTENSIONS")

    embedder = get_embedder()
    store = get_vector_store()
    bm25 = get_bm25_index()

    errors: List[str] = []
    ingested = 0
    skipped = 0

    # We will collect ALL chunks during processing,
    # then build the BM25 index once at the end.
    all_chunks: List[Chunk] = []

    if recreate:
        # Clear existing collection and BM25 index
        store.ensure_collection(vector_size=384, recreate=True)
        bm25.build([])
    else:
        # Ensure collection exists but keep data
        sample_vec = embedder.embed_passages(["dimension_probe"])
        vector_dim = sample_vec.shape[1]
        store.ensure_collection(vector_size=vector_dim, recreate=False)

    # Disable HNSW index during bulk upserts for higher throughput
    store.disable_indexing()

    # For incremental updates (recreate=False), we need to know which documents already exist
    collection_exists = store.collection_exists()

    # Buffer for embedding/upsert batching
    pending_chunks: List[Chunk] = []

    def _flush_batch(chunks_to_process: List[Chunk]) -> int:
        """Embed and upsert a batch of chunks to Qdrant (dense only)."""
        if not chunks_to_process:
            return 0

        batch_texts = [c.text for c in chunks_to_process]
        batch_vectors = embedder.embed_passages(batch_texts)

        store.upsert(chunks_to_process, batch_vectors)
        return len(chunks_to_process)

    try:
        for f in files:
            try:
                logger.debug(f"Processing: {f}")
                doc = parse_document(f)
                doc_id = document_id_for(f)

                # For update runs (recreate=False), delete existing chunks for this document
                if not recreate and collection_exists:
                    store.delete_by_document_id(doc_id)
                    bm25.remove_by_document_id(doc_id)

                file_chunks: List[Chunk] = []
                for block in doc.blocks:
                    if not block.text.strip():
                        continue
                    meta = build_metadata(doc, block)
                    file_chunks.extend(chunk_block(block.text, block.content_type, doc_id, meta))

                if not file_chunks:
                    skipped += 1
                    continue

                # Add to master list and pending batch
                all_chunks.extend(file_chunks)
                pending_chunks.extend(file_chunks)
                ingested += 1

                # Flush to Qdrant whenever batch size is reached
                while len(pending_chunks) >= BATCH_SIZE:
                    batch = pending_chunks[:BATCH_SIZE]
                    pending_chunks = pending_chunks[BATCH_SIZE:]
                    logger.info(f"Embedding & indexing batch of {len(batch)} chunks to Qdrant...")
                    _flush_batch(batch)

            except ExcelliaError as exc:
                logger.error("Failed to ingest %s: %s", f, exc)
                errors.append(f"{f}: {exc}")
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error ingesting %s", f)
                errors.append(f"{f}: {exc}")

        # Flush any remaining chunks to Qdrant
        if pending_chunks:
            logger.info(f"Embedding & indexing remaining {len(pending_chunks)} chunks to Qdrant...")
            _flush_batch(pending_chunks)
            pending_chunks.clear()

        # --------------------------------------------------------------
        # ✅ CRITICAL FIX: Build BM25 index ONCE from all collected chunks
        # --------------------------------------------------------------
        logger.info(f"Building BM25 index from {len(all_chunks)} total chunks...")
        bm25.build(all_chunks)          # this sets self.chunks and creates the BM25Okapi object
        bm25.save()                     # persist to disk

    finally:
        # Re-enable HNSW indexing for fast search after ingestion
        logger.info("Re-enabling Qdrant index...")
        store.enable_indexing()

    logger.info(f"Ingestion completed. Total chunks indexed: {len(all_chunks)}")

    return IngestResponse(
        files_discovered=len(files),
        files_ingested=ingested,
        files_skipped=skipped,
        chunks_indexed=len(all_chunks),
        errors=errors,
    )


def store_safe_delete(store, doc_id: str) -> None:
    try:
        store.delete_by_document_id(doc_id)
    except Exception:  # noqa: BLE001 - collection may not exist yet on first run
        pass