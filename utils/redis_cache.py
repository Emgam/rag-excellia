#!/usr/bin/env python
"""
Redis cache for query responses with TTL and LRU eviction.
"""

import json
import hashlib
import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime

import redis
from config.settings import get_settings

logger = logging.getLogger(__name__)


class QueryCache:
    """Redis-based cache for RAG query responses"""
    
    def __init__(self):
        self.settings = get_settings()
        self.ttl = getattr(self.settings, "QUERY_CACHE_TTL", 3600)
        self.enabled = getattr(self.settings, "ENABLE_QUERY_CACHE", True)
        
        self.client = None
        self.connected = False
        
        if self.enabled:
            try:
                # Try Redis host from environment (Docker) or fallback to localhost
                import os
                redis_host = os.getenv("REDIS_HOST", "localhost")
                redis_port = int(os.getenv("REDIS_PORT", 6379))
                
                self.client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=0,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                self.client.ping()
                self.connected = True
                logger.info(f"✅ Redis cache connected to {redis_host}:{redis_port}")
                print(f"✅ Redis cache connected to {redis_host}:{redis_port}", file=sys.stderr)  # ← DEBUG
            except Exception as e:
                logger.warning(f"⚠️ Redis cache not available: {e}")
                print(f"⚠️ Redis cache not available: {e}", file=sys.stderr)  # ← DEBUG
                self.connected = False
    
    def _get_cache_key(self, question: str, top_k: int) -> str:
        """Generate unique cache key from question and parameters"""
        key_data = f"{question}_{top_k}".encode()
        return f"rag:query:{hashlib.md5(key_data).hexdigest()}"
    
    def get(self, question: str, top_k: int = 3) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        print(f"🔍 [CACHE] get() called for: {question[:40]}...", file=sys.stderr)  # ← DEBUG
        
        if not self.connected or not self.enabled:
            print("❌ [CACHE] Not connected or disabled", file=sys.stderr)  # ← DEBUG
            return None
        
        try:
            cache_key = self._get_cache_key(question, top_k)
            cached = self.client.get(cache_key)
            
            if cached:
                result = json.loads(cached)
                logger.info(f"✅ Cache HIT: {question[:50]}...")
                print(f"✅ [CACHE] HIT: {question[:40]}...", file=sys.stderr)  # ← DEBUG
                self.client.incr("rag:cache:hits", 1)
                self.client.incr("rag:cache:total", 1)
                return result
            else:
                logger.debug(f"❌ Cache MISS: {question[:50]}...")
                print(f"❌ [CACHE] MISS: {question[:40]}...", file=sys.stderr)  # ← DEBUG
                self.client.incr("rag:cache:total", 1)
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Redis get error: {e}")
            print(f"⚠️ [CACHE] get error: {e}", file=sys.stderr)  # ← DEBUG
            return None
    
    def set(self, question: str, top_k: int, result: Dict[str, Any]) -> bool:
        """Cache a response"""
        print(f"💾 [CACHE] set() called for: {question[:40]}...", file=sys.stderr)  # ← DEBUG
        
        if not self.connected or not self.enabled:
            print("❌ [CACHE] Not connected or disabled - cannot set", file=sys.stderr)  # ← DEBUG
            return False
        
        try:
            cache_key = self._get_cache_key(question, top_k)
            
            # Compress result (remove large fields)
            compressed = {
                "answer": result.get("answer", ""),
                "sources": result.get("sources", [])[:3],
                "confidence": result.get("confidence", 0),
                "reliable": result.get("reliable", False),
                "response_time_seconds": result.get("response_time_seconds", 0),
                "cached_at": datetime.now().isoformat()
            }
            
            self.client.setex(cache_key, self.ttl, json.dumps(compressed))
            logger.info(f"✅ Cached: {question[:50]}...")
            print(f"✅ [CACHE] SET successful: {question[:40]}...", file=sys.stderr)  # ← DEBUG
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Redis set error: {e}")
            print(f"⚠️ [CACHE] set error: {e}", file=sys.stderr)  # ← DEBUG
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.connected:
            return {"status": "disconnected"}
        
        try:
            total = int(self.client.get("rag:cache:total") or 0)
            hits = int(self.client.get("rag:cache:hits") or 0)
            keys = len(self.client.keys("rag:query:*"))
            
            hit_rate = f"{hits/total*100:.1f}%" if total > 0 else "0%"
            
            return {
                "status": "connected",
                "total_queries": total,
                "cache_hits": hits,
                "hit_rate": hit_rate,
                "cached_entries": keys,
                "ttl_seconds": self.ttl,
                "enabled": self.enabled
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def clear(self) -> bool:
        """Clear all cache"""
        if not self.connected:
            return False
        
        try:
            keys = self.client.keys("rag:query:*")
            if keys:
                self.client.delete(*keys)
            self.client.delete("rag:cache:hits", "rag:cache:total")
            logger.info("🧹 Cache cleared")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Clear error: {e}")
            return False


# Singleton instance
_cache = None


def get_query_cache() -> QueryCache:
    """Get or create cache instance"""
    global _cache
    if _cache is None:
        _cache = QueryCache()
    return _cache#!/usr/bin/env python
"""
Redis cache for query responses with TTL and LRU eviction.
"""

import json
import hashlib
import logging
from typing import Optional, Dict, Any
from datetime import datetime

import redis
from config.settings import get_settings

logger = logging.getLogger(__name__)


class QueryCache:
    """Redis-based cache for RAG query responses"""
    
    def __init__(self):
        self.settings = get_settings()
        self.ttl = getattr(self.settings, "QUERY_CACHE_TTL", 3600)
        self.enabled = getattr(self.settings, "ENABLE_QUERY_CACHE", True)
        
        self.client = None
        self.connected = False
        
        if self.enabled:
            try:
                # Try Redis host from environment (Docker) or fallback to localhost
                import os
                redis_host = os.getenv("REDIS_HOST", "localhost")
                redis_port = int(os.getenv("REDIS_PORT", 6379))
                
                self.client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=0,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                self.client.ping()
                self.connected = True
                logger.info(f"✅ Redis cache connected to {redis_host}:{redis_port}")
                print(f"✅ Redis cache connected to {redis_host}:{redis_port}", file=sys.stderr)  # ← DEBUG
            except Exception as e:
                logger.warning(f"⚠️ Redis cache not available: {e}")
                print(f"⚠️ Redis cache not available: {e}", file=sys.stderr)  # ← DEBUG
                self.connected = False
    
    def _get_cache_key(self, question: str, top_k: int) -> str:
        """Generate unique cache key from question and parameters"""
        key_data = f"{question}_{top_k}".encode()
        return f"rag:query:{hashlib.md5(key_data).hexdigest()}"
    
    def get(self, question: str, top_k: int = 3) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        print(f"🔍 [CACHE] get() called for: {question[:40]}...", file=sys.stderr)  # ← DEBUG
        
        if not self.connected or not self.enabled:
            print("❌ [CACHE] Not connected or disabled", file=sys.stderr)  # ← DEBUG
            return None
        
        try:
            cache_key = self._get_cache_key(question, top_k)
            cached = self.client.get(cache_key)
            
            if cached:
                result = json.loads(cached)
                logger.info(f"✅ Cache HIT: {question[:50]}...")
                print(f"✅ [CACHE] HIT: {question[:40]}...", file=sys.stderr)  # ← DEBUG
                self.client.incr("rag:cache:hits", 1)
                self.client.incr("rag:cache:total", 1)
                return result
            else:
                logger.debug(f"❌ Cache MISS: {question[:50]}...")
                print(f"❌ [CACHE] MISS: {question[:40]}...", file=sys.stderr)  # ← DEBUG
                self.client.incr("rag:cache:total", 1)
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Redis get error: {e}")
            print(f"⚠️ [CACHE] get error: {e}", file=sys.stderr)  # ← DEBUG
            return None
    
    def set(self, question: str, top_k: int, result: Dict[str, Any]) -> bool:
        """Cache a response"""
        print(f"💾 [CACHE] set() called for: {question[:40]}...", file=sys.stderr)  # ← DEBUG
        
        if not self.connected or not self.enabled:
            print("❌ [CACHE] Not connected or disabled - cannot set", file=sys.stderr)  # ← DEBUG
            return False
        
        try:
            cache_key = self._get_cache_key(question, top_k)
            
            # Compress result (remove large fields)
            compressed = {
                "answer": result.get("answer", ""),
                "sources": result.get("sources", [])[:3],
                "confidence": result.get("confidence", 0),
                "reliable": result.get("reliable", False),
                "response_time_seconds": result.get("response_time_seconds", 0),
                "cached_at": datetime.now().isoformat()
            }
            
            self.client.setex(cache_key, self.ttl, json.dumps(compressed))
            logger.info(f"✅ Cached: {question[:50]}...")
            print(f"✅ [CACHE] SET successful: {question[:40]}...", file=sys.stderr)  # ← DEBUG
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Redis set error: {e}")
            print(f"⚠️ [CACHE] set error: {e}", file=sys.stderr)  # ← DEBUG
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.connected:
            return {"status": "disconnected"}
        
        try:
            total = int(self.client.get("rag:cache:total") or 0)
            hits = int(self.client.get("rag:cache:hits") or 0)
            keys = len(self.client.keys("rag:query:*"))
            
            hit_rate = f"{hits/total*100:.1f}%" if total > 0 else "0%"
            
            return {
                "status": "connected",
                "total_queries": total,
                "cache_hits": hits,
                "hit_rate": hit_rate,
                "cached_entries": keys,
                "ttl_seconds": self.ttl,
                "enabled": self.enabled
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def clear(self) -> bool:
        """Clear all cache"""
        if not self.connected:
            return False
        
        try:
            keys = self.client.keys("rag:query:*")
            if keys:
                self.client.delete(*keys)
            self.client.delete("rag:cache:hits", "rag:cache:total")
            logger.info("🧹 Cache cleared")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Clear error: {e}")
            return False


# Singleton instance
_cache = None


def get_query_cache() -> QueryCache:
    """Get or create cache instance"""
    global _cache
    if _cache is None:
        _cache = QueryCache()
    return _cache