from ingestion.ingest_service import ingest
from retrieval.sparse.bm25_index import get_bm25_index

# Ingest with recreate=True
ingest("test_80", recreate=True)

# Check BM25 index size
bm25 = get_bm25_index()
print(f"BM25 has {len(bm25.chunks)} chunks")
# Expected: 15 (or whatever your total chunk count is)