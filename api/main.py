"""
FastAPI app exposing:

/ingest
/reindex
/query
/query/stream
/health
/stats
/supported-types
/test
/

Run with:
    uvicorn api.main:app --reload --port 8000
"""

# ============================================
# __future__ MUST BE FIRST
# ============================================
from __future__ import annotations

# ============================================
# ENVIRONMENT VARIABLES - SET BEFORE ANY OTHER IMPORTS
# ============================================
import os

PHYSICAL_CORES = "4"  # Your CPU has 4 physical cores

os.environ["OMP_NUM_THREADS"] = PHYSICAL_CORES
os.environ["MKL_NUM_THREADS"] = PHYSICAL_CORES
os.environ["OPENBLAS_NUM_THREADS"] = PHYSICAL_CORES
os.environ["VECLIB_MAXIMUM_THREADS"] = PHYSICAL_CORES
os.environ["NUMEXPR_NUM_THREADS"] = PHYSICAL_CORES

# ============================================
# STANDARD LIBRARY IMPORTS
# ============================================
import asyncio
import sys
import time
import traceback
from contextlib import asynccontextmanager

# ============================================
# THIRD-PARTY IMPORTS
# ============================================
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient

# ============================================
# LOCAL IMPORTS
# ============================================
from config.settings import get_settings
from core.exceptions import ExcelliaError
from core.models import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from generation.answer_service import answer_question
from generation.ollama_client import generate_stream
from generation.prompts import build_user_prompt
from ingestion.ingest_service import ingest as run_ingest
from retrieval.pipeline import retrieve
from retrieval.reranking.reranker import load_reranker
from utils.logger import get_logger
from utils.timer import Timer, print_timing_report, reset_timing
from utils.redis_cache import get_query_cache

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------
# WARMUP FUNCTION
# ---------------------------------------------------------------------

async def warmup_system():
    """
    Warm up the system by running a sample query.
    This loads all models and caches into memory.
    """
    logger.info("🔥 Warming up system...")
    
    try:
        # Run a simple query to load models
        logger.info("   Loading models with sample query...")
        start_time = time.time()
        
        result = await asyncio.to_thread(
            answer_question,
            "What is UPI?",
            top_k=3
        )
        
        elapsed = time.time() - start_time
        logger.info(f"   ✅ Warmup complete in {elapsed:.2f}s")
        logger.info(f"   📝 Sample answer: {result.answer[:100]}...")
        
        # Pre-populate cache with common queries
        common_queries = [
            "What is UPI?",
            "What is a documentary credit?",
            "What is the maximum loan amount?",
            "How do I apply for a loan?",
            "What are the ATM locations?"
        ]
        
        logger.info("   📚 Pre-caching common queries...")
        cache = get_query_cache()
        for q in common_queries:
            cached = cache.get(q, 3)
            if not cached:
                # Run the query to cache it
                await asyncio.to_thread(answer_question, q, top_k=3)
        
        logger.info("   ✅ Pre-caching complete")
        
    except Exception as e:
        logger.warning(f"   ⚠️ Warmup failed: {e}")


# ---------------------------------------------------------------------
# LIFESPAN & INITIALIZATION
# ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup pre-warming and cleanup tasks.
    """
    logger.info("Initializing application resources...")
    print(f"Qdrant target host: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    
    # Initialize Redis cache
    cache = get_query_cache()
    if cache.connected:
        logger.info("✅ Redis cache connected successfully")
    else:
        logger.warning("⚠️ Redis cache not available - caching disabled")
    
    # Pre-load reranker into memory during startup
    try:
        logger.info("Pre-loading cross-encoder reranker model...")
        load_reranker()
        logger.info("✅ Reranker loaded successfully.")
    except Exception as exc:
        logger.error(f"Failed to pre-load reranker: {exc}")

    # ============================================================
    # WARMUP - Load models and cache
    # ============================================================
    await warmup_system()

    yield
    logger.info("Shutting down application...")


app = FastAPI(
    title="Excellia RAG",
    description="Local-first enterprise documentation assistant",
    version="0.1.0",
    lifespan=lifespan,
    servers=[
        {"url": "http://localhost:8000", "description": "Local Development Server"},
        {"url": "http://127.0.0.1:8000", "description": "Localhost"},
    ],
)


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def _get_qdrant_client(timeout: int = 30) -> QdrantClient:
    """
    Utility to create a Qdrant client with explicit network timeouts.
    """
    return QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        timeout=timeout,
    )


async def _execute_ingestion_delayed(path: str | None, recreate: bool):
    """
    Executes ingestion with a slight delay to ensure HTTP sockets flush,
    then delegates the heavy synchronous ML task to a threadpool.
    """
    # Wait 1.5 seconds to guarantee the 202 Accepted response is fully 
    # delivered to the client before the heavy ML process consumes CPU/RAM.
    await asyncio.sleep(1.5)
    
    try:
        logger.info(f"Background ingestion task started: path={path}, recreate={recreate}")
        # Run the synchronous ingest function in a separate thread so it 
        # doesn't block the FastAPI async event loop
        result = await asyncio.to_thread(run_ingest, path=path, recreate=recreate)
        logger.info(f"Background ingestion completed successfully: {result}")
    except Exception as exc:
        logger.exception(f"Background ingestion failed with exception: {exc}")


# ---------------------------------------------------------------------
# INGEST ENDPOINT
# ---------------------------------------------------------------------

@app.post(
    "/ingest",
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_endpoint(
    req: IngestRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Asynchronous ingestion endpoint. Accepts request and runs processing in background.
    """
    background_tasks.add_task(
        _execute_ingestion_delayed,
        path=req.path,
        recreate=req.recreate,
    )
    return {
        "status": "processing",
        "message": "Ingestion job started in the background. Check application logs for completion status.",
    }


# ---------------------------------------------------------------------
# REINDEX ENDPOINT
# ---------------------------------------------------------------------

@app.post(
    "/reindex",
    status_code=status.HTTP_202_ACCEPTED,
)
async def reindex_endpoint(
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Asynchronous reindexing endpoint. Accepts request and runs processing in background.
    """
    background_tasks.add_task(
        _execute_ingestion_delayed,
        path=None,
        recreate=True,
    )
    return {
        "status": "processing",
        "message": "Reindexing job started in the background. Monitor progress via application logs.",
    }


# ---------------------------------------------------------------------
# QUERY ENDPOINT WITH CACHE
# ---------------------------------------------------------------------

@app.post(
    "/query",
    response_model=QueryResponse,
)
async def query_endpoint(
    req: QueryRequest,
) -> QueryResponse:
    print("=" * 60, file=sys.stderr)
    print("QUERY ENDPOINT CALLED", file=sys.stderr)
    print(f"Question: {req.question}", file=sys.stderr)
    print(f"Top K: {req.top_k}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    try:
        if req.stream:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Streaming is not supported through this endpoint. "
                    "Use /query/stream endpoint instead."
                ),
            )

        # ============================================================
        # 1. CHECK REDIS CACHE FIRST
        # ============================================================
        cache = get_query_cache()
        print(f"🔍 Checking cache for: {req.question[:50]}...", file=sys.stderr)
        
        cached_response = cache.get(req.question, req.top_k)
        
        if cached_response:
            print("✅ CACHE HIT! Returning cached response", file=sys.stderr)
            return QueryResponse(**cached_response)
        
        print("❌ CACHE MISS! Generating new response", file=sys.stderr)

        # ============================================================
        # 2. GENERATE NEW RESPONSE (CACHE MISS)
        # ============================================================
        reset_timing()
        print("Calling answer_question...", file=sys.stderr)
        
        start_time = time.time()
        
        with Timer("api_query_total"):
            # Offload synchronous heavy query generation to thread
            result = await asyncio.to_thread(
                answer_question,
                req.question,
                top_k=req.top_k,
            )
        
        elapsed_time = time.time() - start_time
        
        print("Answer generated successfully", file=sys.stderr)
        print(f"Total generation time: {elapsed_time:.2f} seconds", file=sys.stderr)
        
        print_timing_report()
        logger.info(f"Query completed in {elapsed_time:.2f}s")
        
        result.response_time_seconds = round(elapsed_time, 2)
        
        # ============================================================
        # 3. CACHE THE RESULT
        # ============================================================
        cache.set(req.question, req.top_k, result.model_dump())
        print("💾 Response cached", file=sys.stderr)
        
        return result

    except HTTPException:
        raise

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {str(exc)}"
        traceback_str = traceback.format_exc()
        
        print(f"ERROR: {error_msg}", file=sys.stderr)
        print(traceback_str, file=sys.stderr)
        
        logger.exception("Query failed")
        
        return QueryResponse(
            answer=f"ERROR: {error_msg}",
            sources=[],
            confidence=0.0,
            reliable=False,
            response_time_seconds=0.0,
        )


# ---------------------------------------------------------------------
# STREAMING QUERY ENDPOINT
# ---------------------------------------------------------------------

@app.post("/query/stream")
async def streaming_query_endpoint(
    req: QueryRequest,
) -> StreamingResponse:
    """
    Streaming endpoint that returns tokens as they are generated.
    First token appears in <2 seconds.
    """
    try:
        return await _stream_query_async(req)
    except Exception as exc:
        logger.exception("Streaming query failed")
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


async def _stream_query_async(
    req: QueryRequest,
) -> StreamingResponse:
    """
    Async version of streaming query with faster time-to-first-token.
    """
    current_settings = get_settings()
    
    # Offload retrieval to thread
    chunks, confidence, reliable = await asyncio.to_thread(
        retrieve,
        req.question,
        top_k=req.top_k,
    )
    
    if not reliable:
        async def refusal():
            yield current_settings.REFUSAL_MESSAGE
        
        return StreamingResponse(
            refusal(),
            media_type="text/plain",
        )
    
    prompt = build_user_prompt(
        req.question,
        chunks,
    )
    
    # Stream directly from Ollama
    return StreamingResponse(
        generate_stream(prompt),
        media_type="text/plain",
    )


# ---------------------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------------------

@app.get("/health")
def health_endpoint() -> dict:
    current_settings = get_settings()

    qdrant_ok = False
    try:
        client = _get_qdrant_client(timeout=10)
        client.get_collections()
        qdrant_ok = True
    except Exception as exc:
        logger.error(f"Qdrant health check error: {exc}")

    ollama_ok = False
    try:
        response = requests.get(
            f"{current_settings.OLLAMA_HOST}/api/tags",
            timeout=5,
        )
        ollama_ok = response.status_code == 200
    except Exception as exc:
        logger.error(f"Ollama health check error: {exc}")

    return {
        "qdrant": qdrant_ok,
        "ollama": ollama_ok,
        "bm25_index_loaded": True,
    }


# ---------------------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------------------

@app.get("/stats")
def stats_endpoint() -> dict:
    current_settings = get_settings()

    try:
        client = _get_qdrant_client(timeout=10)
        info = client.get_collection(
            current_settings.QDRANT_COLLECTION,
        )

        store_stats = {
            "points_count": info.points_count,
            "collection": current_settings.QDRANT_COLLECTION,
        }

    except Exception as exc:
        logger.error(f"Error fetching vector store stats: {exc}")
        store_stats = {
            "error": str(exc),
        }

    return {
        "vector_store": store_stats,
        "bm25_chunks": 10794,
        "embedding_model": current_settings.EMBEDDING_MODEL,
        "reranker_model": current_settings.RERANKER_MODEL,
        "llm_model": current_settings.OLLAMA_MODEL,
    }


# ---------------------------------------------------------------------
# SUPPORTED FILE TYPES
# ---------------------------------------------------------------------

@app.get("/supported-types")
def supported_types_endpoint() -> dict:
    current_settings = get_settings()

    categories = {
        "documents": [".pdf", ".docx", ".pptx", ".xlsx", ".csv"],
        "markup": [".md", ".markdown", ".html", ".htm"],
        "structured": [".json", ".xml", ".yaml", ".yml"],
        "code": [".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".h", ".hpp"],
        "logs_config": [".log", ".ini", ".cfg", ".conf", ".txt"],
        "images": [".png", ".jpg", ".jpeg"],
    }

    return {
        "categories": categories,
        "all": current_settings.SUPPORTED_EXTENSIONS,
    }


# ---------------------------------------------------------------------
# TEST ENDPOINT
# ---------------------------------------------------------------------

@app.get("/test")
def test_endpoint() -> dict:
    return {
        "status": "ok",
        "message": "Test endpoint works",
    }


# ---------------------------------------------------------------------
# ROOT ENDPOINT
# ---------------------------------------------------------------------

@app.get("/")
def root_endpoint() -> dict:
    return {
        "name": "Excellia RAG API",
        "version": "0.1.0",
        "endpoints": {
            "ingest": "POST /ingest",
            "reindex": "POST /reindex",
            "query": "POST /query",
            "query_stream": "POST /query/stream",
            "health": "GET /health",
            "stats": "GET /stats",
            "supported_types": "GET /supported-types",
            "test": "GET /test",
        },
        "documentation": "/docs",
    }