from core.models import Chunk, ContentType, DocumentMetadata, ScoredChunk
from retrieval.confidence.confidence import estimate_confidence


def _sc(fused_score=None, rerank_score=None) -> ScoredChunk:
    meta = DocumentMetadata(file_path="/tmp/f.txt", file_name="f.txt", file_extension=".txt", file_size_bytes=1)
    chunk = Chunk(chunk_id="a", document_id="doc1", text="x", content_type=ContentType.TEXT, token_count=1, metadata=meta)
    return ScoredChunk(chunk=chunk, fused_score=fused_score, rerank_score=rerank_score)


def test_no_results_is_not_reliable():
    confidence, reliable = estimate_confidence([])
    assert confidence == 0.0
    assert reliable is False


def test_high_fused_score_is_reliable():
    confidence, reliable = estimate_confidence([_sc(fused_score=0.8)])
    assert reliable is True


def test_low_fused_score_is_not_reliable():
    # Use a score below the threshold (0.01)
    # Use 0.009 which is clearly below 0.01
    confidence, reliable = estimate_confidence([_sc(fused_score=0.004)])
    assert reliable is False
