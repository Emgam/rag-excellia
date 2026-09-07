#!/usr/bin/env python
"""
Enhance the validation QA set by adding:
- source_files: list of file names containing the answer (retrieved via pipeline)
- complexity_score: from the smart router
- re-classify: simple/medium/complex based on score
- Deduplicate similar questions.
"""

import json
import os
import sys
from pathlib import Path
from difflib import SequenceMatcher

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from retrieval.pipeline import retrieve
from generation.smart_routing import get_router
from utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
INPUT_PATH = Path("evaluation/validation_qa.json")
OUTPUT_PATH = Path("evaluation/validation_qa_enhanced.json")
SIMILARITY_THRESHOLD = 0.85  # deduplicate questions above this ratio

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def get_source_files(question: str, top_k: int = 8) -> list:
    """Retrieve chunks and extract unique file names."""
    try:
        chunks, _, _ = retrieve(question, top_k=top_k)
        file_names = set()
        for sc in chunks:
            meta = sc.chunk.metadata
            # Try to get file_name from metadata (attribute or dict)
            if hasattr(meta, 'file_name'):
                fname = meta.file_name
            elif hasattr(meta, 'get'):
                fname = meta.get('file_name', '')
            else:
                fname = ''
            if fname:
                file_names.add(fname)
        return list(file_names)
    except Exception as e:
        logger.warning(f"Retrieval failed for '{question[:50]}...': {e}")
        return []

def compute_similarity(a: str, b: str) -> float:
    """Return similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def deduplicate(questions: list) -> list:
    """Remove near-duplicate questions based on similarity threshold."""
    unique = []
    for q in questions:
        # Check if q is too similar to any already kept
        is_dup = False
        for u in unique:
            if compute_similarity(q['question'], u['question']) > SIMILARITY_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            unique.append(q)
    return unique

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    # Load validation set
    if not INPUT_PATH.exists():
        logger.error(f"Input file not found: {INPUT_PATH}")
        return

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} validation questions.")

    # Get router for complexity scoring
    router = get_router()

    enhanced = []
    for idx, item in enumerate(data):
        question = item.get("question", "").strip()
        if not question:
            continue

        logger.info(f"Processing {idx+1}/{len(data)}: {question[:60]}...")

        # 1. Get source files from retrieval
        sources = get_source_files(question)

        # 2. Get complexity score from router
        try:
            complexity_score = router.get_complexity_score(question)
        except Exception as e:
            logger.warning(f"Router failed for '{question[:50]}': {e}")
            complexity_score = 5  # default to medium

        # 3. Map score to class
        if complexity_score <= 3:
            cls = "simple"
        elif complexity_score <= 6:
            cls = "medium"
        else:
            cls = "complex"

        # Keep existing answer and class as fallback, but use computed class
        enhanced_item = {
            "question": question,
            "answer": item.get("answer", ""),
            "class": cls,
            "complexity_score": complexity_score,
            "source_files": sources,
            "original_class": item.get("class", ""),  # for reference
        }
        enhanced.append(enhanced_item)

    # Deduplicate
    before = len(enhanced)
    enhanced = deduplicate(enhanced)
    after = len(enhanced)
    logger.info(f"Deduplication: removed {before - after} duplicate/similar questions.")

    # Save enhanced set
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(enhanced, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Enhanced validation set saved to {OUTPUT_PATH}")
    logger.info(f"   Total: {len(enhanced)} questions.")
    logger.info(f"   Simple: {len([e for e in enhanced if e['class']=='simple'])}")
    logger.info(f"   Medium: {len([e for e in enhanced if e['class']=='medium'])}")
    logger.info(f"   Complex: {len([e for e in enhanced if e['class']=='complex'])}")

    # Print a sample
    if enhanced:
        print("\n🔍 Sample enhanced entry:")
        print(json.dumps(enhanced[0], indent=2))

if __name__ == "__main__":
    main()