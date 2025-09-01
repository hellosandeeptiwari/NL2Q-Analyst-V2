import pytest
from backend.db.engine import PostgresAdapter, SnowflakeAdapter
import os
import time

class DummyConn:
    def cursor(self):
        class DummyCursor:
            def execute(self, sql):
                time.sleep(0.01)
            def fetchall(self):
                return [(1,)]
        return DummyCursor()

@pytest.fixture
def pg_adapter():
    adapter = PostgresAdapter({})
    adapter.conn = DummyConn()
    return adapter

@pytest.fixture
def sf_adapter():
    adapter = SnowflakeAdapter({})
    adapter.conn = DummyConn()
    return adapter

def test_health_latency(pg_adapter):
    health = pg_adapter.health()
    assert health["connected"]
    assert health["latency_ms"] >= 10

def test_health_latency_sf(sf_adapter):
    health = sf_adapter.health()
    assert health["connected"]
    assert health["latency_ms"] >= 10
