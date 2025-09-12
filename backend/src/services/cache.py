"""
Cache Service for NL2Q Analyst V2

Provides intelligent caching for queries, schemas, and results.
"""
import json
import hashlib
from typing import Any, Dict, Optional, List
import structlog

from src.core.config import settings

logger = structlog.get_logger(__name__)


class CacheService:
    """Redis-based cache service with intelligent caching strategies."""
    
    def __init__(self):
        self.redis_client = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize Redis connection."""
        if self._initialized:
            return
        
        try:
            import aioredis
            self.redis_client = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Cache service initialized", redis_url=settings.redis_url)
            self._initialized = True
            
        except Exception as e:
            logger.error("Failed to initialize cache service", error=str(e))
            # Continue without cache
            self.redis_client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.redis_client:
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning("Cache get failed", key=key, error=str(e))
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        if not self.redis_client:
            return False
        
        try:
            ttl = ttl or settings.redis_cache_ttl
            serialized_value = json.dumps(value, default=str)
            
            await self.redis_client.setex(key, ttl, serialized_value)
            return True
            
        except Exception as e:
            logger.warning("Cache set failed", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.warning("Cache delete failed", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.redis_client:
            return False
        
        try:
            return bool(await self.redis_client.exists(key))
        except Exception as e:
            logger.warning("Cache exists check failed", key=key, error=str(e))
            return False
    
    async def ping(self) -> bool:
        """Test cache connection."""
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.ping()
            return True
        except Exception:
            return False
    
    async def cleanup(self):
        """Cleanup cache connection."""
        if self.redis_client:
            await self.redis_client.close()
        self._initialized = False
        logger.info("Cache service cleaned up")
    
    # Specialized cache methods
    
    def _generate_query_cache_key(
        self,
        natural_language: str,
        schema_hash: str,
        tenant_id: str = "default"
    ) -> str:
        """Generate cache key for query results."""
        content = f"{tenant_id}:{natural_language}:{schema_hash}"
        return f"query:{hashlib.md5(content.encode()).hexdigest()}"
    
    def _generate_schema_cache_key(
        self,
        database_id: str,
        tenant_id: str = "default"
    ) -> str:
        """Generate cache key for schema information."""
        return f"schema:{tenant_id}:{database_id}"
    
    async def get_query_cache(
        self,
        natural_language: str,
        schema_hash: str,
        tenant_id: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """Get cached query result."""
        if not settings.enable_query_caching:
            return None
        
        key = self._generate_query_cache_key(natural_language, schema_hash, tenant_id)
        return await self.get(key)
    
    async def set_query_cache(
        self,
        natural_language: str,
        schema_hash: str,
        result: Dict[str, Any],
        tenant_id: str = "default",
        ttl: int = 1800  # 30 minutes default for queries
    ) -> bool:
        """Cache query result."""
        if not settings.enable_query_caching:
            return False
        
        key = self._generate_query_cache_key(natural_language, schema_hash, tenant_id)
        return await self.set(key, result, ttl)
    
    async def get_schema_cache(
        self,
        database_id: str,
        tenant_id: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """Get cached schema information."""
        key = self._generate_schema_cache_key(database_id, tenant_id)
        return await self.get(key)
    
    async def set_schema_cache(
        self,
        database_id: str,
        schema: Dict[str, Any],
        tenant_id: str = "default",
        ttl: int = 3600  # 1 hour default for schemas
    ) -> bool:
        """Cache schema information."""
        key = self._generate_schema_cache_key(database_id, tenant_id)
        return await self.set(key, schema, ttl)
    
    async def invalidate_tenant_cache(self, tenant_id: str) -> int:
        """Invalidate all cache entries for a tenant."""
        if not self.redis_client:
            return 0
        
        try:
            pattern = f"*:{tenant_id}:*"
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                await self.redis_client.delete(*keys)
                logger.info("Invalidated tenant cache", tenant_id=tenant_id, keys_deleted=len(keys))
                return len(keys)
            
            return 0
            
        except Exception as e:
            logger.warning("Failed to invalidate tenant cache", tenant_id=tenant_id, error=str(e))
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.redis_client:
            return {"status": "disabled"}
        
        try:
            info = await self.redis_client.info()
            return {
                "status": "enabled",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "keys": await self.redis_client.dbsize()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Global cache service instance
cache_service = CacheService()