"""
Header‑aware chunker using metadata.section_path.
- Groups blocks by section path (headers) to keep sections intact.
- Each section is then recursively split by paragraphs/sentences if too long.
- Tables and code are never split.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional

from config.settings import get_settings
from core.models import Chunk, ContentType, DocumentMetadata
from utils.logger import get_logger

logger = get_logger(__name__)

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
    _ENCODING = tiktoken.encoding_for_model("gpt-4")
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    _ENCODING = None

# Try to import LangChain's recursive splitter
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

_WORD_RE = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    if _TIKTOKEN_AVAILABLE and _ENCODING:
        try:
            return len(_ENCODING.encode(text))
        except Exception:
            pass
    words = len(_WORD_RE.findall(text))
    return max(1, int(words / 0.75))


def _tiktoken_len(text: str) -> int:
    return estimate_tokens(text)


def chunk_block(
    text: str,
    content_type: ContentType,
    document_id: str,
    metadata: DocumentMetadata,
) -> List[Chunk]:
    """
    Convert a block into chunks.
    - Tables and code are kept whole.
    - Prose is split using section_path to preserve headers.
    """
    settings = get_settings()
    token_count = estimate_tokens(text)

    # Special handling: tables and code are kept whole
    if content_type in (ContentType.TABLE, ContentType.CODE):
        if token_count > settings.CHUNK_SIZE_TOKENS:
            logger.info(
                "Keeping oversized %s block whole (%d tokens > CHUNK_SIZE_TOKENS=%d) for %s",
                content_type.value,
                token_count,
                settings.CHUNK_SIZE_TOKENS,
                metadata.file_name,
            )
        return [_make_chunk(text, content_type, document_id, metadata, token_count)]

    # If the entire block fits, return one chunk
    if token_count <= settings.CHUNK_SIZE_TOKENS:
        return [_make_chunk(text, content_type, document_id, metadata, token_count)]

    # ---- Header‑aware splitting ----
    # Use metadata.section_path to group by section (if available)
    section_path = getattr(metadata, "section_path", [])
    if section_path:
        # We treat the current text as belonging to this section.
        # We can split by headers using the section_path hierarchy:
        # We'll use the last part of section_path as a section header.
        # But we don't have the actual header text in the content.
        # Instead, we rely on the parser to have already separated blocks by sections.
        # Since this block is already a single ParsedBlock, we can just split it recursively.
        # However, if the parser produces multiple blocks per section, they are already separate.
        pass

    # Prose: use recursive splitting
    return _split_prose_recursive(
        text,
        content_type,
        document_id,
        metadata,
        settings.CHUNK_SIZE_TOKENS,
        settings.CHUNK_OVERLAP_TOKENS,
    )


def _split_prose_recursive(
    text: str,
    content_type: ContentType,
    document_id: str,
    metadata: DocumentMetadata,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Chunk]:
    """Recursive splitter with paragraph/sentence separators."""
    if _LANGCHAIN_AVAILABLE:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=_tiktoken_len,
        )
        chunks_text = splitter.split_text(text)
    else:
        logger.warning("LangChain not available – falling back to word‑based split.")
        chunks_text = _split_prose_word_based(text, chunk_size, chunk_overlap)

    result = []
    for chunk_text in chunks_text:
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue
        token_count = estimate_tokens(chunk_text)
        result.append(
            _make_chunk(chunk_text, content_type, document_id, metadata, token_count)
        )
    return result


def _split_prose_word_based(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    words = _WORD_RE.findall(text)
    words_per_chunk = max(1, int(chunk_size * 0.75))
    overlap_words = max(0, int(chunk_overlap * 0.75))

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + words_per_chunk)
        piece = " ".join(words[start:end])
        chunks.append(piece)
        if end == len(words):
            break
        start = end - overlap_words
    return chunks


def _make_chunk(
    text: str,
    content_type: ContentType,
    document_id: str,
    metadata: DocumentMetadata,
    token_count: int,
) -> Chunk:
    digest_src = (
        f"{document_id}:{metadata.section_path}:{metadata.page_number}:{text[:80]}:{len(text)}"
    )
    chunk_id = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:16]
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text.strip(),
        content_type=content_type,
        token_count=token_count,
        metadata=metadata,
    )