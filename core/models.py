from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    CODE = "code"
    IMAGE = "image"


class DocumentMetadata(BaseModel):
    file_path: str
    file_name: str
    file_extension: str
    file_size_bytes: int

    content_type_hint: ContentType = ContentType.TEXT

    title: Optional[str] = None

    section_path: List[str] = Field(
        default_factory=list
    )

    page_number: Optional[int] = None

    language: Optional[str] = None

    extra: Dict[str, Any] = Field(
        default_factory=dict
    )


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str

    content_type: ContentType = ContentType.TEXT

    token_count: int

    metadata: DocumentMetadata


class ScoredChunk(BaseModel):
    chunk: Chunk

    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None

    @property
    def final_score(self) -> float:
        if self.rerank_score is not None:
            return self.rerank_score

        if self.fused_score is not None:
            return self.fused_score

        if self.dense_score is not None:
            return self.dense_score

        if self.sparse_score is not None:
            return self.sparse_score

        return 0.0


class Citation(BaseModel):
    index: int
    document: str
    section: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None



class QueryRequest(BaseModel):
    question: str
    top_k: int = 10
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: List[Citation] = []
    confidence: float = 0.0
    reliable: bool = False
    response_time_seconds: float = 0.0
    retrieved_chunks: Optional[List[ScoredChunk]] = None 


class IngestRequest(BaseModel):
    path: Optional[str] = None
    recreate: bool = False


class IngestResponse(BaseModel):
    files_discovered: int
    files_ingested: int
    files_skipped: int
    chunks_indexed: int

    errors: List[str] = Field(
        default_factory=list
    )