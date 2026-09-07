from core.models import Chunk, ContentType, DocumentMetadata, ScoredChunk
from retrieval.fusion.rrf import rrf_fuse


def _chunk(cid: str) -> Chunk:
    meta = DocumentMetadata(file_path="/tmp/f.txt", file_name="f.txt", file_extension=".txt", file_size_bytes=1)
    return Chunk(chunk_id=cid, document_id="doc1", text=f"text {cid}", content_type=ContentType.TEXT, token_count=2, metadata=meta)


def test_rrf_prefers_items_ranked_high_in_both_lists():
    dense = [ScoredChunk(chunk=_chunk("a"), dense_score=0.9), ScoredChunk(chunk=_chunk("b"), dense_score=0.5)]
    sparse = [ScoredChunk(chunk=_chunk("b"), sparse_score=5.0), ScoredChunk(chunk=_chunk("a"), sparse_score=1.0)]
    fused = rrf_fuse(dense, sparse, k=60, top_n=5)
    ids = [f.chunk.chunk_id for f in fused]
    assert set(ids) == {"a", "b"}
    assert fused[0].fused_score >= fused[1].fused_score


def test_rrf_respects_top_n():
    dense = [ScoredChunk(chunk=_chunk(str(i)), dense_score=1.0 / (i + 1)) for i in range(10)]
    fused = rrf_fuse(dense, [], k=60, top_n=3)
    assert len(fused) == 3
