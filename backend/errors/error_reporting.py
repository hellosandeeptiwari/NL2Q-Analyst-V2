import json
import os
from datetime import datetime
from typing import List, Dict

ERRORS_FILE = "backend/errors/error_reports.jsonl"

def report_error(error: str, context: Dict, user: str = "anonymous"):
    entry = {
        "error": error,
        "context": context,
        "user": user,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(ERRORS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def get_error_reports() -> List[Dict]:
    if not os.path.exists(ERRORS_FILE):
        return []
    with open(ERRORS_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
