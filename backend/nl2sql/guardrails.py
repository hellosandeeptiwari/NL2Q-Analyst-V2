import re
from dataclasses import dataclass

DDL_DML = ("INSERT","UPDATE","DELETE","DROP","ALTER","CREATE","TRUNCATE","MERGE","GRANT","REVOKE")

@dataclass
class GuardrailConfig:
    enable_write: bool
    allowed_schemas: list[str]
    default_limit: int

def sanitize_sql(sql: str, cfg: GuardrailConfig) -> tuple[str,bool]:
    s = sql.strip().strip(";")
    if not cfg.enable_write and s.upper().startswith(DDL_DML):
        raise ValueError("Write operations disabled.")
    # Check for multiple statements (more than one semicolon or semicolon not at end)
    semicolon_count = sql.count(";")
    if semicolon_count > 1 or (semicolon_count == 1 and not sql.strip().endswith(";")):
        raise ValueError("Multiple statements blocked.")
    if "/*" in s or "--" in s:
        # optionally strip comments
        s = re.sub(r"(--.*?$|/\*.*?\*/)", "", s, flags=re.MULTILINE|re.DOTALL)
    # add LIMIT if absent and query seems unbounded
    if re.match(r"^SELECT\b", s, re.I) and re.search(r"\bLIMIT\b", s, re.I) is None:
        s = f"{s} LIMIT {cfg.default_limit}"
        return s, True
    return s, False
