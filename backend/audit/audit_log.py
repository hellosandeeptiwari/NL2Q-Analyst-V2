import json
from datetime import datetime

def log_audit(prompt, sql, execution_time, row_count, error):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt,
        "sql": sql,
        "execution_time": execution_time,
        "row_count": row_count,
        "error": error
    }
    with open("backend/audit/audit_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
