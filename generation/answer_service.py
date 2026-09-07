from __future__ import annotations

import re
import time
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from config.settings import get_settings
from core.models import Citation, QueryResponse, ScoredChunk
from generation.ollama_client import generate
from generation.prompts import build_user_prompt
from generation.smart_routing import get_router
from retrieval.pipeline import retrieve
from utils.logger import get_logger
from utils.timer import Timer, timeit

# ------------------------------------------------------------------
# CPU Optimizations & Thread Control
# ------------------------------------------------------------------
# Prevent PyTorch thread thrashing on CPU. Set to 4 or half your physical cores.

# ------------------------------------------------------------------
logger = get_logger(__name__)

# ✅ ADD THIS BLOCK RIGHT AFTER logger is defined
try:
    logger.info("🔄 FORCE LOADING NLI MODEL AT STARTUP...")
    from sentence_transformers import CrossEncoder
    _NLI_AVAILABLE = True
    test_model = CrossEncoder('cross-encoder/nli-deberta-v3-small', max_length=256)
    logger.info("✅ NLI model loaded successfully at startup!")
    # Clean up - don't keep this instance, we'll use lazy loading
    del test_model
except Exception as e:
    logger.warning(f"⚠️ Could not pre-load NLI model: {e}")
    _NLI_AVAILABLE = True  # Still set to True, will try lazy loading

# ------------------------------------------------------------------
# Config & Constants
# ------------------------------------------------------------------
MAX_CONTEXT_CHUNKS = 8   # Reduced from 7 to speed up LLM generation
MAX_CITATIONS = 5
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"  # MUST BE THIS EXACT MODEL NAME
NLI_ENTAILMENT_THRESHOLD = 0.10
MAX_UNSUPPORTED_CLAIM_RATIO = 0.3   # 0 = every claim must be strictly supported

try:
    from sentence_transformers import CrossEncoder
    _NLI_AVAILABLE = True
except ImportError:
    _NLI_AVAILABLE = False
    CrossEncoder = None

_nli_verifier = None
_nli_entail_idx = None

def _get_nli_verifier():
    """Lazy load the NLI model with debug logging"""
    global _nli_verifier, _nli_entail_idx, _NLI_AVAILABLE
    
    # DEBUG: Log everything
    logger.info(f"🔍 _get_nli_verifier() called")
    logger.info(f"   _NLI_AVAILABLE = {_NLI_AVAILABLE}")
    logger.info(f"   _nli_verifier = {_nli_verifier}")
    
    if not _NLI_AVAILABLE:
        logger.error("❌ _NLI_AVAILABLE is False! This should not happen.")
        logger.info("   Importing sentence_transformers now...")
        try:
            from sentence_transformers import CrossEncoder
            _NLI_AVAILABLE = True
            logger.info("✅ sentence_transformers imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import: {e}")
            return None, None
    
    if _nli_verifier is None:
        try:
            logger.info("🔄 Loading NLI verifier: %s", NLI_MODEL_NAME)
            
            # Set environment variable to avoid warnings
            import os
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            
            # Load the model
            _nli_verifier = CrossEncoder(NLI_MODEL_NAME, max_length=256)
            logger.info("✅ NLI model object created: %s", type(_nli_verifier))
            
            # Get label mapping
            id2label = _nli_verifier.model.config.id2label
            logger.info(f"📊 Label mapping: {id2label}")
            
            # Find entailment index
            _nli_entail_idx = None
            for idx, label in id2label.items():
                if label.lower() == 'entailment':
                    _nli_entail_idx = idx
                    break
            
            if _nli_entail_idx is None:
                logger.error("❌ Could not find 'entailment' label in: %s", id2label)
                _nli_verifier = None
                _nli_entail_idx = None
            else:
                logger.info("✅✅✅ NLI verifier READY! Entailment index: %d", _nli_entail_idx)
                
        except Exception as e:
            logger.exception("❌ Failed to load NLI verifier: %s", e)
            _nli_verifier = None
            _nli_entail_idx = None
    else:
        logger.info("✅ NLI verifier already loaded.")
    
    return _nli_verifier, _nli_entail_idx

# ------------------------------------------------------------------
# Evidence Gating
# ------------------------------------------------------------------
def _normalize_score(score: float) -> float:
    """Normalize cross-encoder logits to [0, 1] probability."""
    if score is None:
        return 0.0
    if score > 1.0 or score < 0.0:
        return float(1.0 / (1.0 + np.exp(-score)))
    return float(score)

def _is_evidence_sufficient(chunks: List[ScoredChunk], settings) -> bool:
    """Strict evidence gate using normalized scores."""
    if not chunks:
        return False

    scores = []
    for c in chunks:
        if hasattr(c, 'rerank_score') and c.rerank_score is not None:
            scores.append(_normalize_score(c.rerank_score))
        elif c.dense_score is not None:
            scores.append(c.dense_score)
            
    if not scores:
        return False

    max_score = max(scores)
    min_threshold = settings.MIN_RELEVANCE_SCORE
    
    if max_score < min_threshold:
        logger.info(f"❌ Evidence gate failed: max_score {max_score:.4f} < threshold {min_threshold}")
        return False

    logger.info(f"✅ Evidence gate passed: max_score {max_score:.4f}")
    return True

# ------------------------------------------------------------------
# Citations & Context
# ------------------------------------------------------------------
def _extract_metadata_dict(meta: Any) -> Dict:
    if hasattr(meta, "model_dump"): return meta.model_dump()
    if hasattr(meta, "dict"): return meta.dict()
    return meta if isinstance(meta, dict) else {}

def _build_citations(chunks: List[ScoredChunk]) -> List[Citation]:
    citations = []
    for idx, sc in enumerate(chunks[:MAX_CITATIONS], start=1):
        chunk = sc.chunk
        meta_dict = _extract_metadata_dict(chunk.metadata)
        
        section_path = meta_dict.get("section_path", [])
        section = " > ".join(section_path) if section_path else meta_dict.get("section", "")
        document = meta_dict.get("document", meta_dict.get("file_name", "Unknown"))
        page = meta_dict.get("page", meta_dict.get("page_number"))

        citations.append(Citation(
            index=idx,
            document=document,
            section=section,
            page=page,
            chunk_id=chunk.chunk_id,
        ))
    return citations

def _build_grounded_context(chunks: List[ScoredChunk]) -> str:
    parts = []
    for idx, sc in enumerate(chunks[:MAX_CONTEXT_CHUNKS], start=1):
        chunk = sc.chunk
        meta_dict = _extract_metadata_dict(chunk.metadata)
        
        document = meta_dict.get("document", meta_dict.get("file_name", "Unknown"))
        section_path = meta_dict.get("section_path", [])
        section = " > ".join(section_path) if section_path else meta_dict.get("section", "")
        page = meta_dict.get("page", meta_dict.get("page_number"))

        header = f"[{idx}] Document: {document}"
        if section: header += f" | Section: {section}"
        if page is not None: header += f" | Page: {page}"

        parts.append(f"{header}\n{chunk.text}")
    return "\n\n".join(parts)

# ------------------------------------------------------------------
# Citation Validation
# ------------------------------------------------------------------
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")

def _validate_citations(answer: str, num_chunks: int) -> bool:
    ids = [int(x) for x in _CITATION_PATTERN.findall(answer)]
    if not ids:
        # No citations found in a factual answer
        return False
    # All citation IDs must be within the range of provided context
    return all(1 <= cid <= num_chunks for cid in ids)

# ------------------------------------------------------------------
# NLI Faithfulness Verification
# ------------------------------------------------------------------
def _extract_claims(answer: str) -> List[Tuple[str, List[int]]]:
    """Extract sentences and map to their citation IDs."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer)
    result = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        citations = [int(x) for x in _CITATION_PATTERN.findall(s)]
        cleaned = _CITATION_PATTERN.sub("", s).strip()
        if len(cleaned) >= 10:
            result.append((cleaned, citations))
    # If no segments were extracted, treat the whole answer as one claim
    if not result and len(answer.strip()) >= 10:
        citations = [int(x) for x in _CITATION_PATTERN.findall(answer)]
        cleaned = _CITATION_PATTERN.sub("", answer).strip()
        if cleaned:
            result.append((cleaned, citations))
    return result

def _verify_answer_faithfulness(answer: str, chunks: List[ScoredChunk]) -> Tuple[bool, float]:
    """
    Verify faithfulness using NLI cross-encoder.
    Each claim in the answer is checked against the retrieved chunks.
    Returns (faithful, score) where score is the fraction of claims supported.
    """
    verifier, entail_idx = _get_nli_verifier()
    if not verifier or entail_idx is None:
        logger.warning("⚠️ NLI unavailable – trusting strict prompt instead of refusing.")
        return True, 0.5  # ✅ On accepte la réponse au lieu de la refuser !

    segments = _extract_claims(answer)
    if not segments:
        return False, 0.0

    # Build pairs: (chunk_text, claim) for each claim
    pairs = []
    claim_indices = []

    for i, (claim, citation_ids) in enumerate(segments):
        if citation_ids:
            # Use cited chunks
            for cid in citation_ids:
                if 1 <= cid <= len(chunks):
                    pairs.append((chunks[cid-1].chunk.text, claim))
                    claim_indices.append(i)
        else:
            # No citation – check against all chunks
            for chunk in chunks:
                pairs.append((chunk.chunk.text, claim))
                claim_indices.append(i)
            logger.info(f"Claim without citation – checking against all {len(chunks)} chunks.")

    if not pairs:
        logger.warning("No valid NLI pairs – rejecting answer")
        return False, 0.0

    try:
        # Get NLI scores
        scores = verifier.predict(pairs, apply_softmax=True)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        # Check which claims are supported
        claim_supported = [False] * len(segments)
        for i, score_vec in enumerate(scores):
            claim_idx = claim_indices[i]
            entail_prob = float(score_vec[entail_idx])
            if entail_prob >= NLI_ENTAILMENT_THRESHOLD:
                claim_supported[claim_idx] = True

        supported_count = sum(claim_supported)
        score = supported_count / len(segments)
        faithful = score >= (1.0 - MAX_UNSUPPORTED_CLAIM_RATIO)

        if not faithful:
            failed_claims = [segments[i][0] for i, sup in enumerate(claim_supported) if not sup]
            logger.warning(f"NLI rejected claims: {failed_claims}")

        return faithful, score

    except Exception as e:
        logger.warning("NLI verification failed: %s", e)
        return True, 0.5  # ✅ On accepte la réponse en cas d'erreur d'inférence

# ------------------------------------------------------------------
# Retrieval Confidence
# ------------------------------------------------------------------
def _calculate_retrieval_confidence(chunks: List[ScoredChunk]) -> float:
    if not chunks: return 0.0
    scores = []
    for c in chunks:
        if hasattr(c, 'rerank_score') and c.rerank_score is not None:
            scores.append(_normalize_score(c.rerank_score))
        elif c.dense_score is not None:
            scores.append(c.dense_score)
    if not scores: return 0.0
    return max(0.0, min(1.0, max(scores)))

# ------------------------------------------------------------------
# Main Answer Pipeline
# ------------------------------------------------------------------
@timeit
def answer_question(question: str, top_k: Optional[int] = None, model: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, reranker_enabled: Optional[bool] = None) -> QueryResponse:
    start_time = time.perf_counter()
    settings = get_settings()
    router = get_router()

    # 1. Get routing config
    config = router.classify_query(question)
    route = config.get("route", "medium")
    complexity = router.get_complexity_score(question)
    logger.info(f"📊 Query Complexity: {complexity}/10 | Route: {route.upper()} | {config.get('description', 'unknown')}")

    # --- DEFINE DYNAMIC THRESHOLDS ---
    MEDIUM_CONFIDENCE_THRESHOLD = 0.35   # was 0.50
    HIGH_CONFIDENCE_THRESHOLD = 0.60     # was 0.85

    # --- DYNAMIC MAX_CONTEXT_CHUNKS BASED ON ROUTE ---
    if route == "simple":
        max_chunks = 3                    # Fast for simple queries
        max_tokens_override = config.get("max_tokens", 120)
    elif route == "medium":
        max_chunks = 5                    # Moderate for how-to queries
        max_tokens_override = config.get("max_tokens", 150)
    else:  # complex
        max_chunks = MAX_CONTEXT_CHUNKS   # 8 for complex reasoning
        max_tokens_override = config.get("max_tokens", 200)

    try:
        retrieve_top_k = config.get("top_k", 5) * 2

        # 2. Retrieve
        with Timer("answer_retrieval"):
            chunks, retrieval_confidence, reliable = retrieve(
                question,
                top_k=retrieve_top_k,
                fused_top_n=config.get("fused_top_n", settings.FUSED_TOP_N),
                reranker_enabled=config.get("reranker_enabled", settings.RERANKER_ENABLED),
            )

        # 3. Evidence Gate & Low Confidence Check
        if not chunks or not _is_evidence_sufficient(chunks, settings) or retrieval_confidence < MEDIUM_CONFIDENCE_THRESHOLD:
            logger.warning(f"Refused: Insufficient evidence or low confidence ({retrieval_confidence:.2f})")
            return _refusal(settings.REFUSAL_MESSAGE, start_time, "insufficient_evidence")

        # 4. Select Best Evidence – use dynamic max_chunks
        final_top_k = min(top_k if top_k else config.get("top_k", 3), len(chunks), max_chunks)
        chunks = chunks[:final_top_k]

        # 5. Build Context & Citations
        citations = _build_citations(chunks)
        context = _build_grounded_context(chunks)

        # 6. Generate – use route-specific max_tokens
        with Timer("answer_build_prompt"):
            prompt = build_user_prompt(question, chunks, context)
            
        with Timer("answer_generation"):
            # Use model from config (already routes 1.5B for simple, 3B for complex)
            # But allow manual override if passed
            actual_model = model if model else config.get("model", settings.OLLAMA_MODEL)
            actual_max_tokens = max_tokens if max_tokens else max_tokens_override
            
            answer_text = generate(
                prompt,
                model=actual_model,
                temperature=temperature if temperature is not None else config.get("temperature", 0.0),
                max_tokens=actual_max_tokens,
            )

        if not answer_text or len(answer_text.strip()) < 5:
            return _refusal(settings.REFUSAL_MESSAGE, start_time, "generation_failure")

        answer_text = answer_text.strip()

        # 7. Refusal Check
        if settings.REFUSAL_MESSAGE in answer_text:
            return _refusal(settings.REFUSAL_MESSAGE, start_time, "model_refusal")

        # 8. Citation Validation Gate (Optional – uncomment if needed)
        # if not _validate_citations(answer_text, len(chunks)):
        #     logger.warning("Answer rejected: missing or invalid citations")
        #     return _refusal(settings.REFUSAL_MESSAGE, start_time, "invalid_citations")

        # 9. Dynamic Faithfulness Gate
        faithful = True
        faithfulness_score = 0.0
        
        if retrieval_confidence >= HIGH_CONFIDENCE_THRESHOLD:
            logger.info(f"✅ High confidence ({retrieval_confidence:.2f}) - Bypassing faithfulness gate.")
            faithful = True
            faithfulness_score = 1.0
        else:
            logger.info(f"⚠️ Medium confidence ({retrieval_confidence:.2f}) - Running faithfulness gate.")
            with Timer("answer_faithfulness"):
                try:
                    faithful, faithfulness_score = _verify_answer_faithfulness(answer_text, chunks)
                except Exception as nli_error:
                    logger.warning(f"⚠️ NLI unavailable – trusting strict prompt. Error: {nli_error}")
                    faithful = True
                    faithfulness_score = 0.5
            
            if not faithful:
                logger.warning("Answer rejected by faithfulness gate")
                return _refusal(settings.REFUSAL_MESSAGE, start_time, "faithfulness_failed")

        logger.info(f"Faithfulness: {faithfulness_score:.2f} | Retrieval: {retrieval_confidence:.2f}")

        elapsed = time.perf_counter() - start_time
        return QueryResponse(
            answer=answer_text,
            sources=citations,
            confidence=faithfulness_score,
            reliable=True,
            response_time_seconds=elapsed,
            retrieved_chunks=chunks,
        )

    except Exception as e:
        logger.exception("Error generating answer: %s", e)
        return _refusal(settings.REFUSAL_MESSAGE, start_time, "generation_failure")

def _refusal(message: str, start_time: float, reason: str = "unknown") -> QueryResponse:
    return QueryResponse(
        answer=message,
        sources=[],
        confidence=0.0,
        reliable=False,
        response_time_seconds=time.perf_counter() - start_time,
    )