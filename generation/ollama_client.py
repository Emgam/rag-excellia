"""HTTP client for Ollama's /api/chat endpoint. Tries OLLAMA_MODEL first,
falls back to OLLAMA_FALLBACK_MODEL on failure/timeout."""
from __future__ import annotations

from typing import Iterator, Optional

import requests

from config.settings import get_settings
from core.exceptions import GenerationError
from utils.logger import get_logger

logger = get_logger(__name__)

# Import the strict production system prompt from prompts.py
try:
    from generation.prompts import SYSTEM_PROMPT
except ImportError:
    # Fallback just in case prompts.py is missing during development
    SYSTEM_PROMPT = "You are a strict documentation assistant. Answer ONLY using the provided context. Cite facts with [1]."
    logger.warning("⚠️ SYSTEM_PROMPT not found in prompts.py, using basic fallback.")


def _chat(
    model: str,
    user_prompt: str,
    stream: bool,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> requests.Response:
    """Send a chat request to Ollama and return the response."""
    settings = get_settings()
    
    # Use provided values or fallback to settings
    temp = temperature if temperature is not None else settings.OLLAMA_TEMPERATURE
    max_tok = max_tokens if max_tokens is not None else settings.OLLAMA_MAX_TOKENS
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": stream,
        "options": {
            "temperature": temp,
            "num_predict": max_tok,
        },
    }
    
    logger.debug(f"Sending request to Ollama with model: {model}, temperature: {temp}")
    
    return requests.post(
        f"{settings.OLLAMA_HOST}/api/chat",
        json=payload,
        timeout=settings.OLLAMA_TIMEOUT_SECONDS,
    )


def generate(
    user_prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a non-streaming response from Ollama."""
    settings = get_settings()
    
    primary_model = model if model else settings.OLLAMA_MODEL
    fallback_model = settings.OLLAMA_FALLBACK_MODEL
    
    logger.info(f"🤖 Generating with model: {primary_model}")
    
    models_to_try = [primary_model, fallback_model]
    
    for idx, model_name in enumerate(models_to_try):
        if not model_name:
            continue
            
        try:
            resp = _chat(
                model_name,
                user_prompt,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            
            if content:
                if idx > 0:
                    logger.info(f"✅ Success with fallback model: {model_name}")
                return content
            else:
                logger.warning(f"Empty response from model: {model_name}")
                
        except requests.exceptions.Timeout:
            logger.warning(f"⏰ Timeout with model {model_name}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"🔌 Connection error with model {model_name}")
        except Exception as exc:
            logger.warning(f"❌ Model {model_name} failed: {exc}")
    
    raise GenerationError(
        "All configured Ollama models failed. Is `ollama serve` running and are the models pulled? "
        f"Tried: {', '.join([m for m in models_to_try if m])}"
    )


def generate_stream(
    user_prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Iterator[str]:
    """Generate a streaming response from Ollama."""
    import json as _json
    
    settings = get_settings()
    
    primary_model = model if model else settings.OLLAMA_MODEL
    fallback_model = settings.OLLAMA_FALLBACK_MODEL
    
    models_to_try = [primary_model, fallback_model]
    resp = None
    
    for idx, model_name in enumerate(models_to_try):
        if not model_name:
            continue
            
        try:
            resp = _chat(
                model_name,
                user_prompt,
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            resp.raise_for_status()
            if idx > 0:
                logger.info(f"🔄 Streaming with fallback model: {model_name}")
            break
        except Exception as exc:
            logger.warning(f"Streaming with {model_name} failed: {exc}")
            if idx == len(models_to_try) - 1:
                raise GenerationError(f"All models failed for streaming: {exc}")
    
    if resp is None:
        raise GenerationError("No response from Ollama")
    
    logger.info(f"📡 Streaming generation with {primary_model}")
    
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            chunk = _json.loads(line)
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content
            if chunk.get("done"):
                break
        except _json.JSONDecodeError:
            continue
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            continue