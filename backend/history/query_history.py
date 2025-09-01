import json
import os
from typing import List, Dict

HISTORY_FILE = "backend/history/query_history.json"

def save_query_history(nl: str, sql: str, job_id: str, user: str = "anonymous"):
    history = load_query_history()
    entry = {
        "nl": nl,
        "sql": sql,
        "job_id": job_id,
        "user": user,
        "timestamp": json.dumps({"timestamp": "now"})  # Placeholder
    }
    history.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f)

def load_query_history() -> List[Dict]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def get_recent_queries(limit: int = 10) -> List[Dict]:
    history = load_query_history()
    return history[-limit:]
