#!/usr/bin/env python3

import requests
import json

def test_api_query():
    """Test the API endpoint with debug logging"""
    
    url = "http://localhost:8000/api/agent/query"
    
    payload = {
        "query": "read table final nba output python and fetch top 5 rows",
        "user_id": "default_user",
        "session_id": "test_session"
    }
    
    print(f"🔍 Testing API query with payload: {payload}")
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Query successful")
            print(f"📋 Plan status: {result.get('status', 'unknown')}")
            
            # Check task results
            if 'task_results' in result:
                for task_id, task_result in result['task_results'].items():
                    if 'execution' in task_id.lower():
                        print(f"🔍 Execution task {task_id}: {task_result.get('status', 'unknown')}")
                        if 'data' in task_result:
                            print(f"📊 Data rows: {len(task_result['data'])}")
        else:
            print(f"❌ Query failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Request error: {e}")

if __name__ == "__main__":
    test_api_query()
