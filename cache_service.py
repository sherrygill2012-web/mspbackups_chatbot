"""
Cache Service for MSP360 RAG Chatbot
Provides caching for embeddings and search results to reduce latency and API costs
Supports both in-memory and Redis backends
"""

import os
import json
import hashlib
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from functools import wraps
import threading

from dotenv import load_dotenv

load_dotenv()

# Check if Redis is available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class CacheEntry:
    """Cache entry with value and metadata"""
    value: Any
    timestamp: float
    ttl: float
    hit_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() - self.timestamp > self.ttl
    
    def touch(self):
        """Update hit count"""
        self.hit_count += 1


class RedisCache:
    """
    Redis-based cache with TTL support.
    Provides persistent caching across restarts.
    """
    
    def __init__(
        self,
        url: str = None,
        prefix: str = "msp360",
        default_ttl: float = 3600,  # 1 hour default
        max_size: int = None  # Not used for Redis, included for interface compatibility
    ):
        """
        Initialize Redis cache.
        
        Args:
            url: Redis URL (e.g., redis://192.168.1.146:6379)
            prefix: Key prefix for namespacing
            default_ttl: Default time-to-live in seconds
            max_size: Not used (Redis handles eviction)
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis package required. Install with: pip install redis")
        
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.max_size = max_size  # For interface compatibility
        
        # Parse URL and connect
        self.client = redis.from_url(
            self.url,
            decode_responses=False,  # We handle encoding ourselves
            socket_timeout=5,
            socket_connect_timeout=5
        )
        
        # Test connection
        try:
            self.client.ping()
            self._connected = True
        except redis.ConnectionError as e:
            print(f"Warning: Redis connection failed: {e}")
            self._connected = False
        
        # Local stats (Redis doesn't track these)
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }
    
    def _make_key(self, key: str) -> str:
        """Create namespaced key"""
        return f"{self.prefix}:{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if not self._connected:
            self.stats["misses"] += 1
            return None
        
        try:
            full_key = self._make_key(key)
            data = self.client.get(full_key)
            
            if data is None:
                self.stats["misses"] += 1
                return None
            
            self.stats["hits"] += 1
            return json.loads(data)
            
        except Exception as e:
            print(f"Redis get error: {e}")
            self.stats["misses"] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        if not self._connected:
            return
        
        try:
            full_key = self._make_key(key)
            ttl_seconds = int(ttl or self.default_ttl)
            data = json.dumps(value)
            self.client.setex(full_key, ttl_seconds, data)
        except Exception as e:
            print(f"Redis set error: {e}")
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted, False if not found
        """
        if not self._connected:
            return False
        
        try:
            full_key = self._make_key(key)
            return self.client.delete(full_key) > 0
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False
    
    def clear(self):
        """Clear all cache entries with our prefix"""
        if not self._connected:
            return
        
        try:
            pattern = f"{self.prefix}:*"
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=100)
                if keys:
                    self.client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            print(f"Redis clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        
        # Try to get Redis info
        size = 0
        if self._connected:
            try:
                pattern = f"{self.prefix}:*"
                cursor, keys = self.client.scan(0, match=pattern, count=1000)
                size = len(keys)
            except:
                pass
        
        return {
            **self.stats,
            "size": size,
            "backend": "redis",
            "connected": self._connected,
            "url": self.url.split("@")[-1] if "@" in self.url else self.url,  # Hide password
            "hit_rate": f"{hit_rate:.2%}"
        }


class InMemoryCache:
    """
    Thread-safe in-memory cache with TTL support.
    Supports LRU eviction when max size is reached.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 3600,  # 1 hour default
        cleanup_interval: float = 300  # 5 minutes
    ):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
            cleanup_interval: How often to clean expired entries
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments"""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    def _cleanup_if_needed(self):
        """Remove expired entries if cleanup interval has passed"""
        if time.time() - self._last_cleanup < self.cleanup_interval:
            return
        
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
                self.stats["expirations"] += 1
            
            self._last_cleanup = time.time()
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self._cache:
            return
        
        # Find entry with lowest hit count (simple LRU approximation)
        lru_key = min(
            self._cache.keys(),
            key=lambda k: (self._cache[k].hit_count, self._cache[k].timestamp)
        )
        del self._cache[lru_key]
        self.stats["evictions"] += 1
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        self._cleanup_if_needed()
        
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self.stats["misses"] += 1
                return None
            
            if entry.is_expired():
                del self._cache[key]
                self.stats["expirations"] += 1
                self.stats["misses"] += 1
                return None
            
            entry.touch()
            self.stats["hits"] += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                self._evict_lru()
            
            self._cache[key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl
            )
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total if total > 0 else 0
            
            return {
                **self.stats,
                "size": len(self._cache),
                "max_size": self.max_size,
                "backend": "memory",
                "hit_rate": f"{hit_rate:.2%}"
            }


def create_cache(
    cache_type: str = None,
    redis_url: str = None,
    prefix: str = "msp360",
    max_size: int = 1000,
    default_ttl: float = 3600
):
    """
    Factory function to create appropriate cache backend.
    
    Args:
        cache_type: "redis" or "memory" (auto-detects if not specified)
        redis_url: Redis URL for Redis backend
        prefix: Key prefix for Redis
        max_size: Max size for in-memory cache
        default_ttl: Default TTL in seconds
    
    Returns:
        Cache instance (RedisCache or InMemoryCache)
    """
    cache_type = cache_type or os.getenv("CACHE_BACKEND", "auto")
    redis_url = redis_url or os.getenv("REDIS_URL")
    
    # Auto-detect: use Redis if URL is provided and redis package is available
    if cache_type == "auto":
        if redis_url and REDIS_AVAILABLE:
            cache_type = "redis"
        else:
            cache_type = "memory"
    
    if cache_type == "redis":
        if not REDIS_AVAILABLE:
            print("Warning: Redis requested but not available, falling back to memory")
            return InMemoryCache(max_size=max_size, default_ttl=default_ttl)
        return RedisCache(url=redis_url, prefix=prefix, default_ttl=default_ttl)
    
    return InMemoryCache(max_size=max_size, default_ttl=default_ttl)


class EmbeddingCache:
    """
    Specialized cache for embeddings.
    Uses content hash as key for deduplication.
    """
    
    def __init__(
        self,
        max_size: int = 5000,
        ttl: float = 86400  # 24 hours for embeddings
    ):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum number of cached embeddings
            ttl: Time-to-live for embeddings
        """
        self.cache = create_cache(
            prefix="msp360:emb",
            max_size=max_size,
            default_ttl=ttl
        )
    
    def _hash_text(self, text: str, model: str = "") -> str:
        """Generate hash for text content"""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def get(self, text: str, model: str = "") -> Optional[List[float]]:
        """
        Get cached embedding.
        
        Args:
            text: Text that was embedded
            model: Model used for embedding
            
        Returns:
            Embedding vector or None
        """
        key = self._hash_text(text, model)
        return self.cache.get(key)
    
    def set(self, text: str, embedding: List[float], model: str = ""):
        """
        Cache an embedding.
        
        Args:
            text: Text that was embedded
            embedding: Embedding vector
            model: Model used for embedding
        """
        key = self._hash_text(text, model)
        self.cache.set(key, embedding)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()


class SearchResultCache:
    """
    Cache for search results.
    Caches based on query, category, and limit.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: float = 1800  # 30 minutes for search results
    ):
        """
        Initialize search result cache.
        
        Args:
            max_size: Maximum number of cached searches
            ttl: Time-to-live for search results
        """
        self.cache = create_cache(
            prefix="msp360:search",
            max_size=max_size,
            default_ttl=ttl
        )
    
    def _generate_key(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5
    ) -> str:
        """Generate cache key for search parameters"""
        key_data = json.dumps({
            "query": query.lower().strip(),
            "category": category,
            "limit": limit
        }, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    def get(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5
    ) -> Optional[List[Dict]]:
        """
        Get cached search results.
        
        Args:
            query: Search query
            category: Category filter
            limit: Number of results
            
        Returns:
            List of search results or None
        """
        key = self._generate_key(query, category, limit)
        return self.cache.get(key)
    
    def set(
        self,
        query: str,
        results: List[Dict],
        category: Optional[str] = None,
        limit: int = 5
    ):
        """
        Cache search results.
        
        Args:
            query: Search query
            results: Search results to cache
            category: Category filter
            limit: Number of results
        """
        key = self._generate_key(query, category, limit)
        self.cache.set(key, results)
    
    def invalidate_category(self, category: str):
        """
        Invalidate all cache entries for a category.
        Note: This is a simplified version that clears all cache.
        For production, you'd want to track category->key mappings.
        """
        # For simplicity, we don't implement category-specific invalidation
        # In production, you'd maintain a reverse index
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()


# Global cache instances
_embedding_cache: Optional[EmbeddingCache] = None
_search_cache: Optional[SearchResultCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """Get or create the global embedding cache"""
    global _embedding_cache
    if _embedding_cache is None:
        max_size = int(os.getenv("EMBEDDING_CACHE_SIZE", "5000"))
        ttl = float(os.getenv("EMBEDDING_CACHE_TTL", "86400"))
        _embedding_cache = EmbeddingCache(max_size=max_size, ttl=ttl)
    return _embedding_cache


def get_search_cache() -> SearchResultCache:
    """Get or create the global search result cache"""
    global _search_cache
    if _search_cache is None:
        max_size = int(os.getenv("SEARCH_CACHE_SIZE", "1000"))
        ttl = float(os.getenv("SEARCH_CACHE_TTL", "1800"))
        _search_cache = SearchResultCache(max_size=max_size, ttl=ttl)
    return _search_cache


def cached_embedding(func):
    """
    Decorator to cache embedding function results.
    
    Usage:
        @cached_embedding
        async def generate_embedding(text: str, model: str) -> List[float]:
            ...
    """
    @wraps(func)
    async def wrapper(self, text: str, *args, **kwargs):
        cache = get_embedding_cache()
        model = getattr(self, 'model', '')
        
        # Check cache
        cached = cache.get(text, model)
        if cached is not None:
            return cached
        
        # Generate and cache
        result = await func(self, text, *args, **kwargs)
        cache.set(text, result, model)
        return result
    
    return wrapper


def cached_search(func):
    """
    Decorator to cache search function results.
    
    Note: This decorator is currently unused. Caching is implemented
    directly in QdrantTools.search_docs() for more control.
    
    Usage:
        @cached_search
        async def search_docs(query: str, category: str = None, limit: int = 5):
            ...
    """
    @wraps(func)
    async def wrapper(self, query: str, category: Optional[str] = None, limit: int = 5, *args, **kwargs):
        # Skip cache if reranking is disabled or bypass_cache is True
        use_reranking = kwargs.get('use_reranking', True)
        bypass_cache = kwargs.get('bypass_cache', False)
        
        if not use_reranking or bypass_cache:
            return await func(self, query, category, limit, *args, **kwargs)
        
        cache = get_search_cache()
        
        # Check cache
        cached = cache.get(query, category, limit)
        if cached is not None:
            return cached
        
        # Search and cache
        result = await func(self, query, category, limit, *args, **kwargs)
        cache.set(query, result, category, limit)
        return result
    
    return wrapper


def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all caches"""
    return {
        "embedding_cache": get_embedding_cache().get_stats(),
        "search_cache": get_search_cache().get_stats()
    }


def clear_all_caches():
    """Clear all caches"""
    get_embedding_cache().cache.clear()
    get_search_cache().cache.clear()
