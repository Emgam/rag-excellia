from functools import lru_cache

@lru_cache(maxsize=100)
def memory_cache_get(key: str):
    # Super fast in-memory cache
    pass