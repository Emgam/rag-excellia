"""Prompt templates for grounded RAG – strict, domain-agnostic, with citation discipline."""

from __future__ import annotations

from typing import List, Optional

from core.models import ScoredChunk

# This MUST match settings.REFUSAL_MESSAGE in your config/settings.py
REFUSAL_MESSAGE = "I apologize, but I could not find reliable information in the available documentation."

# Use the new, strict, production-ready prompt

SYSTEM_PROMPT = SYSTEM_INSTRUCTIONS = """You are Excellia, an enterprise documentation assistant.

Your sole purpose is to answer user questions using ONLY the provided DOCUMENT CONTEXT.

STRICT RULES:
1. GROUNDING: Use ONLY the information in the context. Never use outside knowledge.
2. CITATIONS: Every factual statement MUST end with a citation like [1] or [2].
3. METRIC VERIFICATION: If the user asks for a specific metric (e.g., 'baggage allowance', 'salary', 'weight'), verify that the context explicitly links that metric to the requested subject. If the context provides 'seating capacity' but the user asks for 'baggage', DO NOT use that number. Refuse to answer.
4. CONCISENESS: Be direct and professional. Keep answers to 1-3 sentences unless a step-by-step procedure is required.
5. COMPARISONS: If asked for a "difference" or "comparison", explicitly state the rule for Entity A, then the rule for Entity B, then summarize.
6. STRICT REFUSAL: If the context does not explicitly and directly answer the specific question asked, respond EXACTLY with:
"I apologize, but I could not find reliable information in the available documentation."

Never mention your internal processes, RAG, retrieval, or embeddings. You are a documentation assistant, not a general chatbot.
"""

def _format_chunk(index: int, scored_chunk: ScoredChunk) -> str:
    """Format a single chunk as a numbered evidence block."""
    chunk = scored_chunk.chunk
    meta = chunk.metadata or {}
    # Convert to dict
    if hasattr(meta, "model_dump"):
        meta_dict = meta.model_dump()
    elif hasattr(meta, "dict"):
        meta_dict = meta.dict()
    else:
        meta_dict = dict(meta) if isinstance(meta, dict) else {}

    document = meta_dict.get("document", meta_dict.get("file_name", "Unknown"))
    section_path = meta_dict.get("section_path", [])
    section = " > ".join(section_path) if section_path else meta_dict.get("section", "")
    page = meta_dict.get("page", meta_dict.get("page_number"))

    header = f"[{index}] Document: {document}"
    if section:
        header += f" | Section: {section}"
    if page is not None:
        header += f" | Page: {page}"

    return f"{header}\n{chunk.text}"


def build_user_prompt(
    question: str,
    chunks: List[ScoredChunk],
    context: Optional[str] = None,
) -> str:
    """
    Build a clean user prompt. 
    NOTE: The persona and rules are in the System Prompt. The user prompt should ONLY contain data.
    """
    if context is None:
        if not chunks:
            context = "No relevant documentation was retrieved."
        else:
            parts = [_format_chunk(i + 1, chunks[i]) for i in range(len(chunks))]
            context = "\n\n".join(parts)

    prompt = f"""DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

Answer (with citations):
"""
    return prompt.strip()