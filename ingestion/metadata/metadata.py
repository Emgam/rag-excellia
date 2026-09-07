"""Builds DocumentMetadata for each parsed block: file info + section
hierarchy + page number, so citations can point back to something a
human can actually locate in the source file."""
from __future__ import annotations

from pathlib import Path

from core.models import ContentType, DocumentMetadata
from ingestion.parsers.parser import ParsedBlock, ParsedDocument


def build_metadata(doc: ParsedDocument, block: ParsedBlock) -> DocumentMetadata:
    stat = doc.file_path.stat()
    return DocumentMetadata(
        file_path=str(doc.file_path),
        file_name=doc.file_path.name,
        file_extension=doc.file_path.suffix.lower(),
        file_size_bytes=stat.st_size,
        content_type_hint=block.content_type,
        title=doc.title,
        section_path=block.section_path,
        page_number=block.page_number,
        extra=block.extra,
    )


def document_id_for(path: Path) -> str:
    return str(path.resolve())
