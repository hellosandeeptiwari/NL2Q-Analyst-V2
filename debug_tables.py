#!/usr/bin/env python3
"""
Quick test to check what data is actually in the tables
"""

import requests
import json

def test_table_contents():
    queries_to_test = [
        "SELECT * FROM VOLUME LIMIT 5",
        "SELECT * FROM METRICS LIMIT 5", 
        "SELECT COUNT(*) as volume_count FROM VOLUME",
        "SELECT COUNT(*) as metrics_count FROM METRICS",
        "SELECT DISTINCT NPI FROM VOLUME LIMIT 10",
        "SELECT DISTINCT PROVIDER_NAME FROM METRICS LIMIT 10"
    ]
    
    for query in queries_to_test:
        try:
            response = requests.post(
                'http://localhost:8000/api/agent/query',
                json={
                    'query': f'Execute this SQL: {query}',
                    'user_id': 'debug_user', 
                    'session_id': 'debug_session'
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                # Find execution results
                execution_steps = {k: v for k, v in result.get('results', {}).items() if 'execution' in k}
                
                for step_name, step_data in execution_steps.items():
                    if step_data.get('results'):
                        print(f"\n📊 Query: {query}")
                        print(f"   Results: {step_data['results']}")
                    else:
                        print(f"\n❌ Query: {query}")
                        print(f"   No results returned")
            else:
                print(f"❌ API Error for '{query}': {response.status_code}")
                
        except Exception as e:
            print(f"❌ Request failed for '{query}': {e}")

if __name__ == "__main__":
    test_table_contents()
