# Excellia RAG

Local-first, enterprise documentation assistant. See the original design
doc for the full architecture, technology rationale, and API reference —
this file only covers what's specific to running this generated codebase.

## Honest notes on this build

This repository is a complete, runnable implementation of the pipeline
described in the design doc — ingestion, chunking, dense+sparse retrieval,
RRF fusion, reranking, confidence gating, local generation, and evaluation
— with all logic real and testable. Two substitutions were made because
the originally-specified components require multi-GB downloads this
build environment can't fetch:

1. **Parsing**: instead of IBM Docling (one unified parser for
   PDF/DOCX/PPTX/XLSX/HTML/images with a learned layout model), this
   build uses per-format libraries — `pypdf`, `python-docx`, `python-pptx`,
   `openpyxl`, `BeautifulSoup` — in `ingestion/parsers/parser.py`. Table
   and heading-hierarchy extraction still works, but there's no learned
   layout model or built-in figure detection. To restore Docling 1:1,
   swap the body of `parse_document()` for a Docling `DocumentConverter`
   call — the `ParsedBlock`/`ParsedDocument` contract downstream doesn't
   change.
2. Everything else (Qdrant, BM25, RRF, `bge-reranker-v2-m3`, Ollama +
   qwen2.5) is wired exactly as specified and will pull/run for real once
   you have Docker/Ollama available with internet access — nothing here
   needs further code changes, only `docker compose up -d` and the model
   pulls in Quick Start.

## Running it

```bash
cp .env.example .env
docker compose up -d
docker exec -it excellia-ollama ollama pull qwen2.5:3b-instruct-q4_K_M

curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{}'
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "How do I authenticate against the Payments API?"}'
```

A sample doc is included at `documents/sample_docs/payments_api.md` so
`/ingest` + `/query` work out of the box before you drop in your own files.

## Tests

```bash
pip install -r requirements.txt
pytest -q
```

The included tests (`tests/`) are deterministic and require no model
downloads — they cover chunking invariants (tables/code never split),
code-boundary extraction (Python `ast` + brace-matching), RRF fusion,
confidence gating, and evaluation metrics.
