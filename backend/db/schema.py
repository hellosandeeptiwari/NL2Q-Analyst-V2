import os
import json
from .engine import get_adapter

SCHEMA_CACHE_PATH = "backend/db/schema_cache.json"


def get_schema_cache():
    if os.path.exists(SCHEMA_CACHE_PATH):
        with open(SCHEMA_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    adapter = get_adapter()
    allowlist = os.getenv("ALLOWED_SCHEMAS", "public").split(",")
    schema = adapter.get_schema_snapshot(allowlist)
    with open(SCHEMA_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f)
    return schema
