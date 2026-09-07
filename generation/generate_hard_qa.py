#!/usr/bin/env python
"""
Generate a hard QA dataset with multi‑hop questions from the document corpus.
Uses the LLM to create questions that require combining information from
multiple sources or sections.

Output: evaluation/qa_hard.json
"""

import json
import random
from pathlib import Path
from typing import List, Dict

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Project imports
from config.settings import get_settings
from ingestion.discovery.discover import discover_files
from ingestion.parsers.parser import parse_document
from ingestion.chunking.chunker import chunk_block
from ingestion.metadata.metadata import build_metadata, document_id_for
from utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DOCS_ROOT = Path("./documents")            # your full corpus
OUTPUT_PATH = Path("evaluation/qa_hard.json")
NUM_QUESTIONS = 150                        # total questions to generate
QUESTIONS_PER_BATCH = 5                    # per LLM prompt
SAMPLE_CHUNKS_PER_PROMPT = 10              # chunks to include in context
MAX_FILES = 0                              # 0 = all files; set e.g., 200 for faster test

# ------------------------------------------------------------------
# Load documents and split into chunks
# ------------------------------------------------------------------
def load_chunks_from_docs(root: Path, max_files: int = 0) -> List[Dict]:
    """
    Parse all supported files in root and return a list of chunk dicts.
    Each dict: {'text': str, 'file_name': str, 'doc_id': str}
    """
    files = discover_files(root)
    if max_files:
        files = files[:max_files]
    logger.info(f"Found {len(files)} files, processing...")

    all_chunks = []
    for f in files:
        try:
            doc = parse_document(f)
            doc_id = document_id_for(f)
            file_name = f.name

            for block in doc.blocks:
                if not block.text.strip():
                    continue
                # Build a proper metadata object (has .section_path, etc.)
                meta = build_metadata(doc, block)
                chunks = chunk_block(block.text, block.content_type, doc_id, meta)
                for c in chunks:
                    all_chunks.append({
                        "text": c.text,
                        "file_name": file_name,
                        "doc_id": doc_id,
                    })
        except Exception as e:
            logger.warning(f"Error parsing {f}: {e}")
            continue
    logger.info(f"Loaded {len(all_chunks)} chunks.")
    return all_chunks

# ------------------------------------------------------------------
# Sample chunks for context
# ------------------------------------------------------------------
def sample_chunks_for_prompt(chunks: List[Dict], n: int) -> List[Dict]:
    """Return a random sample of n chunks, ensuring diversity by file."""
    # Group by file
    file_groups = {}
    for c in chunks:
        file_groups.setdefault(c["file_name"], []).append(c)
    # Pick from different files if possible
    selected = []
    file_list = list(file_groups.keys())
    random.shuffle(file_list)
    for f in file_list:
        if len(selected) >= n:
            break
        group = file_groups[f]
        take = min(len(group), n - len(selected))
        selected.extend(random.sample(group, take))
    # If still short, fill from remaining
    if len(selected) < n:
        remaining = [c for c in chunks if c not in selected]
        extra = random.sample(remaining, min(len(remaining), n - len(selected)))
        selected.extend(extra)
    return selected

# ------------------------------------------------------------------
# Prompt the LLM to generate multi‑hop questions
# ------------------------------------------------------------------
def generate_questions_from_chunks(chunks: List[Dict], llm) -> List[Dict]:
    """Given a list of chunks, prompt the LLM to produce multi-hop Q&A."""
    # Build context with source file labels
    context = ""
    for i, c in enumerate(chunks, 1):
        context += f"[Source {i}: {c['file_name']}]\n{c['text']}\n\n"

    prompt_template = PromptTemplate.from_template("""
You are given a set of document excerpts from various files.
Your task is to generate {num_questions} complex, multi‑hop questions that CANNOT be answered by a single excerpt.
Each question must require combining information from at least TWO different excerpts (sources).

Rules:
- Questions should be about the content (e.g., differences, relationships, step‑by‑step processes, comparisons).
- Each question must be at least 15 words long.
- For each question, provide:
  - The question string.
  - A correct answer that synthesizes information from the referenced sources.
  - A list of source file names (from the excerpts) that contain the evidence (at least 2).

Excerpts:
{context}

Return a JSON list of exactly {num_questions} objects, each with keys: "question", "answer", "source_files".
Only return valid JSON, no additional text.
""")

    prompt = prompt_template.format(
        num_questions=QUESTIONS_PER_BATCH,
        context=context
    )

    response = llm.invoke(prompt)
    # Remove markdown fences if present
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.endswith("```"):
        response = response[:-3]
    try:
        data = json.loads(response)
        if isinstance(data, list):
            return data
        else:
            logger.warning("Response is not a list: %s", response[:200])
            return []
    except json.JSONDecodeError as e:
        logger.warning("JSON decode error: %s\nResponse: %s", e, response[:300])
        return []

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    # Load chunks from the documents folder
    chunks = load_chunks_from_docs(DOCS_ROOT, max_files=MAX_FILES)
    if not chunks:
        logger.error("No chunks loaded. Check DOCS_ROOT path.")
        return

    # Initialize LLM (uses the same model as your generation)
    settings = get_settings()
    model_name = getattr(settings, "LLM_MODEL", "qwen2.5:3b-instruct-q4_k_m")
    llm = OllamaLLM(model=model_name, temperature=0.3)

    all_qa = []
    attempts = 0
    max_attempts = 100  # safety

    while len(all_qa) < NUM_QUESTIONS and attempts < max_attempts:
        attempts += 1
        logger.info(f"Generation attempt {attempts}, collected {len(all_qa)}/{NUM_QUESTIONS}")

        # Sample chunks for context
        sample = sample_chunks_for_prompt(chunks, SAMPLE_CHUNKS_PER_PROMPT)
        if not sample:
            break

        batch = generate_questions_from_chunks(sample, llm)
        if not batch:
            continue

        # Validate: each must have required fields and at least 2 source_files
        for item in batch:
            if not all(k in item for k in ("question", "answer", "source_files")):
                continue
            if not isinstance(item["source_files"], list) or len(item["source_files"]) < 2:
                continue
            # Ensure source_files are strings
            item["source_files"] = [str(s) for s in item["source_files"]]
            all_qa.append(item)
            if len(all_qa) >= NUM_QUESTIONS:
                break

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_qa[:NUM_QUESTIONS], f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Saved {len(all_qa[:NUM_QUESTIONS])} hard QA questions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()