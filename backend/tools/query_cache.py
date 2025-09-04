"""
Query Cache - Result caching and lineage tracking
Provides fast retrieval, cache invalidation, and lineage metadata for audit and observability
"""

import hashlib
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class CacheEntry:
    cache_key: str
    query: str
    result: Any
    created_at: datetime
    expires_at: Optional[datetime]
    lineage: Dict[str, Any]
    user_id: str
    hit_count: int = 0
    last_accessed: datetime = None

class QueryCache:
    """
    Enterprise-grade query result cache with lineage tracking
    """
    def __init__(self):
        self.cache: Dict[str, CacheEntry] = {}
        self.default_ttl = 3600  # seconds
    
    def _generate_cache_key(self, query: str, user_id: str) -> str:
        key_str = f"{query}|{user_id}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, user_id: str) -> Optional[CacheEntry]:
        cache_key = self._generate_cache_key(query, user_id)
        entry = self.cache.get(cache_key)
        if entry and (not entry.expires_at or entry.expires_at > datetime.now()):
            entry.hit_count += 1
            entry.last_accessed = datetime.now()
            return entry
        return None
    
    def set(self, query: str, result: Any, user_id: str, lineage: Dict[str, Any], ttl: Optional[int] = None):
        cache_key = self._generate_cache_key(query, user_id)
        expires_at = datetime.now() + timedelta(seconds=ttl or self.default_ttl)
        entry = CacheEntry(
            cache_key=cache_key,
            query=query,
            result=result,
            created_at=datetime.now(),
            expires_at=expires_at,
            lineage=lineage,
            user_id=user_id,
            hit_count=1,
            last_accessed=datetime.now()
        )
        self.cache[cache_key] = entry
    
    def invalidate(self, query: str, user_id: str):
        cache_key = self._generate_cache_key(query, user_id)
        if cache_key in self.cache:
            del self.cache[cache_key]
    
    def get_lineage(self, query: str, user_id: str) -> Optional[Dict[str, Any]]:
        entry = self.get(query, user_id)
        if entry:
            return entry.lineage
        return None
    
    def clear_expired(self):
        now = datetime.now()
        expired_keys = [key for key, entry in self.cache.items() if entry.expires_at and entry.expires_at < now]
        for key in expired_keys:
            del self.cache[key]
    
    def stats(self) -> Dict[str, Any]:
        return {
            "total_entries": len(self.cache),
            "active_entries": sum(1 for entry in self.cache.values() if not entry.expires_at or entry.expires_at > datetime.now()),
            "hits": sum(entry.hit_count for entry in self.cache.values()),
            "last_accessed": max((entry.last_accessed for entry in self.cache.values() if entry.last_accessed), default=None)
        }
