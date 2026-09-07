from core.models import ContentType, DocumentMetadata
from ingestion.chunking.chunker import chunk_block, estimate_tokens


def _meta(name="f.txt"):
    return DocumentMetadata(file_path=f"/tmp/{name}", file_name=name, file_extension=".txt", file_size_bytes=10)


def test_short_text_is_single_chunk():
    text = "This is a short piece of text."
    chunks = chunk_block(text, ContentType.TEXT, "doc1", _meta())
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_long_text_is_split_with_overlap():
    text = " ".join([f"word{i}" for i in range(2000)])
    chunks = chunk_block(text, ContentType.TEXT, "doc1", _meta())
    assert len(chunks) > 1
    for c in chunks:
        assert estimate_tokens(c.text) <= 600  # size + slack


def test_table_never_split_even_if_oversized():
    huge_table = "| a | b |\n| --- | --- |\n" + "\n".join([f"| {i} | val{i} |" for i in range(5000)])
    chunks = chunk_block(huge_table, ContentType.TABLE, "doc1", _meta("t.csv"))
    assert len(chunks) == 1
    assert chunks[0].content_type == ContentType.TABLE


def test_code_block_never_split():
    code = "def f():\n" + "\n".join([f"    x{i} = {i}" for i in range(500)]) + "\n    return x0"
    chunks = chunk_block(code, ContentType.CODE, "doc1", _meta("f.py"))
    assert len(chunks) == 1
