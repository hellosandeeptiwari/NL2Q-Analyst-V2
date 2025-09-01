import pytest
from backend.nl2sql.guardrails import sanitize_sql, GuardrailConfig

def test_block_ddl_dml():
    cfg = GuardrailConfig(enable_write=False, allowed_schemas=["public"], default_limit=100)
    for op in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "MERGE", "GRANT", "REVOKE"]:
        with pytest.raises(ValueError):
            sanitize_sql(f"{op} table_name", cfg)

def test_block_multi_statement():
    cfg = GuardrailConfig(enable_write=False, allowed_schemas=["public"], default_limit=100)
    with pytest.raises(ValueError):
        sanitize_sql("SELECT * FROM users; SELECT * FROM orders", cfg)

def test_limit_injection():
    cfg = GuardrailConfig(enable_write=False, allowed_schemas=["public"], default_limit=100)
    sql, added = sanitize_sql("SELECT * FROM users", cfg)
    assert "LIMIT 100" in sql
    assert added

def test_no_limit_if_present():
    cfg = GuardrailConfig(enable_write=False, allowed_schemas=["public"], default_limit=100)
    sql, added = sanitize_sql("SELECT * FROM users LIMIT 50", cfg)
    assert "LIMIT 50" in sql
    assert not added
