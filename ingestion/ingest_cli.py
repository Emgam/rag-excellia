#!/usr/bin/env python
"""Direct ingestion script – bypasses the API for debugging."""
import sys
sys.path.insert(0, '.')

from pathlib import Path
from ingestion.ingest_service import ingest

if __name__ == "__main__":
    # Use the exact same path you send to the API
    path = "/app/documents"
    result = ingest(path=path, recreate=False)
    print(f"Files discovered: {result.files_discovered}")
    print(f"Files ingested: {result.files_ingested}")
    print(f"Chunks indexed: {result.chunks_indexed}")
    print(f"Errors: {result.errors}")