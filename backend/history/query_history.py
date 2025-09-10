import json
import os
from typing import List, Dict, Any
import datetime

HISTORY_FILE = "backend/history/query_history.json"

def save_query_history(nl: str, sql: str, job_id: str, user: str = "anonymous", results: List[Dict] = None, columns: List[str] = None):
    """Save query history with optional results and metadata"""
    history = load_query_history()
    
    # Limit results size for storage efficiency 
    sample_data = []
    row_count = 0
    if results:
        row_count = len(results)
        sample_data = results[:5]  # Store first 5 rows as sample
    
    entry = {
        "nl": nl,
        "sql": sql,
        "job_id": job_id,
        "user": user,
        "timestamp": datetime.datetime.now().isoformat(),
        "results": sample_data,  # Store sample data
        "columns": columns or [],
        "row_count": row_count
    }
    
    history.append(entry)
    
    # Keep only last 50 entries to prevent file growth
    if len(history) > 50:
        history = history[-50:]
    
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def load_query_history() -> List[Dict]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []

def get_recent_queries(limit: int = 10, user: str = None) -> List[Dict]:
    """Get recent queries, optionally filtered by user"""
    history = load_query_history()
    
    if user:
        history = [q for q in history if q.get('user') == user]
    
    return history[-limit:]

def get_last_query_with_data(user: str = None) -> Dict[str, Any]:
    """Get the most recent query that has actual data results"""
    history = load_query_history()
    
    if user:
        history = [q for q in history if q.get('user') == user]
    
    # Find most recent query with data
    for query in reversed(history):
        if query.get('results') and len(query['results']) > 0:
            return query
    
    return {}
