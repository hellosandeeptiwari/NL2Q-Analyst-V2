import pytest
import os
import json
from backend.db.schema import get_schema_cache

SCHEMA_CACHE_PATH = "backend/db/schema_cache.json"

class DummyAdapter:
    def get_schema_snapshot(self, allowlist):
        return {"users": {"id": "int", "name": "text"}}


def test_schema_cache_warmup(monkeypatch):
    monkeypatch.setattr("backend.db.engine.get_adapter", lambda: DummyAdapter())
    if os.path.exists(SCHEMA_CACHE_PATH):
        os.remove(SCHEMA_CACHE_PATH)
    schema = get_schema_cache()
    assert "users" in schema
    assert os.path.exists(SCHEMA_CACHE_PATH)
    with open(SCHEMA_CACHE_PATH) as f:
        cached = json.load(f)
    assert cached == schema
