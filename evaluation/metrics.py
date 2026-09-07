"""
Retrieval metrics (Precision@K, Recall@K, MRR, NDCG) against a golden
query -> relevant-chunk-ID dataset, plus lexical-overlap generation
metrics (faithfulness, citation accuracy, hallucination rate, answer
relevance). The generation metrics are intentionally NOT LLM-judge
based -- everything here is local heuristics, in keeping with the
project's "no cloud APIs" design goal.
"""
from __future__ import annotations

import math
import re
from typing import List, Set

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> Set[str]:
    return {t.lower() for t in _WORD_RE.findall(text)}


# --- Retrieval metrics ------------------------------------------------------

def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    return len([r for r in top if r in relevant_ids]) / len(top)


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    return len(top & relevant_ids) / len(relevant_ids)


def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    top = retrieved_ids[:k]
    dcg = sum((1.0 / math.log2(i + 2)) for i, cid in enumerate(top) if cid in relevant_ids)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum((1.0 / math.log2(i + 2)) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# --- Generation metrics (lexical-overlap heuristics) ------------------------

def faithfulness(answer: str, context: str) -> float:
    """Fraction of non-trivial answer tokens that also appear in the
    supplied context — a cheap proxy for 'did the model stick to the
    source material' without calling out to an external LLM judge."""
    a_tokens = _tokens(answer)
    c_tokens = _tokens(context)
    a_tokens = {t for t in a_tokens if len(t) > 2}
    if not a_tokens:
        return 1.0
    return len(a_tokens & c_tokens) / len(a_tokens)


def citation_accuracy(answer: str, num_available_citations: int) -> float:
    """Fraction of citation markers like [1], [2] in the answer that
    reference an index that actually exists in the provided context."""
    cited = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    if not cited:
        return 0.0
    valid = [c for c in cited if 1 <= c <= num_available_citations]
    return len(valid) / len(cited)


def hallucination_rate(answer: str, context: str) -> float:
    return 1.0 - faithfulness(answer, context)


def answer_relevance(answer: str, question: str) -> float:
    a_tokens = _tokens(answer)
    q_tokens = {t for t in _tokens(question) if len(t) > 2}
    if not q_tokens:
        return 0.0
    return len(a_tokens & q_tokens) / len(q_tokens)
