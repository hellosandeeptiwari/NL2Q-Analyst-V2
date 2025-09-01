import json
import os
from datetime import datetime
from typing import Dict

ANALYTICS_FILE = "backend/analytics/usage.json"

def log_usage(endpoint: str, user: str = "anonymous"):
    analytics = load_analytics()
    date = datetime.utcnow().date().isoformat()
    if date not in analytics:
        analytics[date] = {}
    if endpoint not in analytics[date]:
        analytics[date][endpoint] = 0
    analytics[date][endpoint] += 1
    with open(ANALYTICS_FILE, "w", encoding="utf-8") as f:
        json.dump(analytics, f)

def load_analytics() -> Dict:
    if os.path.exists(ANALYTICS_FILE):
        with open(ANALYTICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def get_usage_stats() -> Dict:
    return load_analytics()
