import re
from dataclasses import dataclass

DDL_DML = ("INSERT","UPDATE","DELETE","DROP","ALTER","CREATE","TRUNCATE","MERGE","GRANT","REVOKE")

@dataclass
class GuardrailConfig:
    enable_write: bool
    allowed_schemas: list[str]
    default_limit: int

def sanitize_sql(sql: str, cfg: GuardrailConfig) -> tuple[str,bool]:
    """Basic safety checks only - no modification of LLM-generated SQL"""
    print(f"🧠 LLM Generated SQL (passed through unmodified):\n{sql}")
    s = sql.strip().strip(";")
    
    # Basic safety checks only
    if not cfg.enable_write and s.upper().startswith(DDL_DML):
        raise ValueError("Write operations disabled.")
    
    # Check for multiple statements (security check)
    semicolon_count = sql.count(";")
    if semicolon_count > 1 or (semicolon_count == 1 and not sql.strip().endswith(";")):
        raise ValueError("Multiple statements blocked.")
    
    # Return SQL unmodified - LLM with complete semantic analysis should generate correct syntax
    return s, False
