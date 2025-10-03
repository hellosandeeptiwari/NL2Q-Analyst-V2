import re
from dataclasses import dataclass

DDL_DML = ("INSERT","UPDATE","DELETE","DROP","ALTER","CREATE","TRUNCATE","MERGE","GRANT","REVOKE")

@dataclass
class GuardrailConfig:
    enable_write: bool
    allowed_schemas: list[str]
    default_limit: int

def sanitize_sql(sql: str, cfg: GuardrailConfig) -> tuple[str,bool]:
    """Enhanced SQL cleaning with markdown removal and safety checks"""
    print(f"🧠 Raw LLM Response:\n{sql}")
    
    # 🔧 CRITICAL FIX: Clean markdown code blocks that cause syntax errors
    cleaned_sql = sql.strip()
    
    # Remove markdown code blocks (```sql ... ``` or ``` ... ```)
    if cleaned_sql.startswith('```'):
        # Find the start of actual SQL (after the opening ```)
        lines = cleaned_sql.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('```'):
                start_idx = i + 1
                break
        
        # Find the end of SQL (before closing ```)
        end_idx = len(lines)
        for i in range(len(lines) - 1, start_idx - 1, -1):
            if lines[i].strip() == '```' or lines[i].strip().startswith('```'):
                end_idx = i
                break
        
        # Extract clean SQL
        if start_idx < len(lines):
            cleaned_sql = '\n'.join(lines[start_idx:end_idx]).strip()
            print(f"🧹 Cleaned markdown blocks - extracted SQL:\n{cleaned_sql}")
    
    # Additional cleaning
    s = cleaned_sql.strip().strip(";")
    
    # Basic safety checks
    if not cfg.enable_write and s.upper().startswith(DDL_DML):
        raise ValueError("Write operations disabled.")
    
    # Check for multiple statements (security check)
    semicolon_count = s.count(";")
    if semicolon_count > 1 or (semicolon_count == 1 and not s.strip().endswith(";")):
        raise ValueError("Multiple statements blocked.")
    
    print(f"✅ Final cleaned SQL:\n{s}")
    return s, False
