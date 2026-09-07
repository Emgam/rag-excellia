"""
Centralized, .env-driven configuration for Excellia RAG.

All configurable values are defined here and can be overridden through
environment variables or the .env file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide configuration loaded from environment variables
    and the .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Paths & Relevance Gating
    # ------------------------------------------------------------------

    DOCUMENTS_DIR: str = "documents"
    DATA_DIR: str = "data"
    MIN_RELEVANCE_SCORE: float = 0.35
    MIN_REQUIRED_CHUNKS: int = 1
    
    # ------------------------------------------------------------------
    # Document Ingestion & Chunking (Updated to 350/50 sweet spot)
    # ------------------------------------------------------------------

    SUPPORTED_EXTENSIONS: list[str] = Field(
        default_factory=lambda: [
            ".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".md", ".markdown",
            ".html", ".htm", ".txt", ".json", ".xml", ".yaml", ".yml",
            ".log", ".ini", ".cfg", ".conf", ".py", ".js", ".ts",
            ".java", ".go", ".rs", ".cpp", ".c", ".h", ".hpp",
            ".png", ".jpg", ".jpeg",
        ]
    )

    MAX_FILE_SIZE_MB: int = 100
    CHUNK_SIZE_TOKENS: int = 250
    CHUNK_OVERLAP_TOKENS: int = 0

    # ------------------------------------------------------------------
    # Embeddings & Vector Store
    # ------------------------------------------------------------------

    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 32

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_COLLECTION: str = "excellia_docs"
    QDRANT_ON_DISK: bool = True
    QDRANT_TIMEOUT: int = 120

    # ------------------------------------------------------------------
    # Hybrid Retrieval (Sparse + Dense + RRF) - Updated for CPU Speed
    # ------------------------------------------------------------------

    BM25_INDEX_PATH: str = "data/bm25_index.pkl"
    DENSE_TOP_K: int = 20
    SPARSE_TOP_K: int = 20
    FUSED_TOP_N: int = 50
    RRF_K: int = 30

    # ------------------------------------------------------------------
    # Cross-encoder Reranking & Gating (Updated to True, RERANK_TOP_K=4)
    # ------------------------------------------------------------------
    ENABLE_QUERY_CACHE: bool = True
    QUERY_CACHE_TTL: int = 3600
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_K: int = 4

    CONFIDENCE_THRESHOLD: float = 0.01
    REFUSAL_MESSAGE: str = (
        "I could not find reliable information in the available documentation."
    )

    # ------------------------------------------------------------------
    # LLM — Ollama
    # ------------------------------------------------------------------

    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b-instruct-q4_k_m"
    OLLAMA_FALLBACK_MODEL: str = "qwen2.5:3b-instruct-q4_k_m"
    OLLAMA_TIMEOUT_SECONDS: int = 600
    OLLAMA_TEMPERATURE: float = 0.0
    OLLAMA_MAX_TOKENS: int = 300
    OLLAMA_KEEP_ALIVE: str = "-1"
    
    OMP_NUM_THREADS: int = 4
    MKL_NUM_THREADS: int = 4
    OPENBLAS_NUM_THREADS: int = 4

    # Under Cache & Memory
    ENABLE_MEMORY_CACHE: bool = True
    MEMORY_CACHE_SIZE: int = 100
    
    # ------------------------------------------------------------------
    # Cache & Processing
    # ------------------------------------------------------------------

    ASYNC_PROCESSING: bool = False
    STREAMING_ENABLED: bool = False

    # ------------------------------------------------------------------
    # SMART ROUTING - QUERY CLASSIFICATION (Updated for CPU Speed)
    # ------------------------------------------------------------------

    # Simple queries (definition, factual)
    SIMPLE_MODEL: str = "qwen2.5:1.5b-instruct-q4_k_m"
    SIMPLE_TOP_K: int = 3
    SIMPLE_MAX_TOKENS: int = 80
    SIMPLE_RERANKER: bool = False
    SIMPLE_CHUNK_SIZE: int = 200
    SIMPLE_TEMPERATURE: float = 0.0

    # Medium queries (how-to, process)
    MEDIUM_MODEL: str = "qwen2.5:1.5b-instruct-q4_k_m"
    MEDIUM_TOP_K: int = 4
    MEDIUM_MAX_TOKENS: int = 120
    MEDIUM_RERANKER: bool = True
    MEDIUM_CHUNK_SIZE: int = 250
    MEDIUM_TEMPERATURE: float = 0.0

    # Complex queries (comparison, analysis)
    COMPLEX_MODEL: str = "qwen2.5:3b-instruct-q4_k_m"
    COMPLEX_TOP_K: int = 3
    COMPLEX_MAX_TOKENS: int = 120
    COMPLEX_RERANKER: bool = True
    COMPLEX_CHUNK_SIZE: int = 300
    COMPLEX_TEMPERATURE: float = 0.0

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # ------------------------------------------------------------------
    # Computed Paths
    # ------------------------------------------------------------------

    @property
    def documents_path(self) -> Path:
        path = Path(self.DOCUMENTS_DIR).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def data_path(self) -> Path:
        path = Path(self.DATA_DIR).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()