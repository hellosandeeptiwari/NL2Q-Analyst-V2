#!/usr/bin/env python3
"""
Quick test to check if the API is returning data correctly
"""

import requests
import json

try:
    # Test with a simple query
    response = requests.post(
        'http://localhost:8000/api/agent/query',
        json={
            'query': 'show me simple test data',
            'user_id': 'test_user', 
            'session_id': 'test_session'
        },
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ API Response Success!")
        print(f"Status: {result.get('status')}")
        print(f"Results keys: {list(result.get('results', {}).keys())}")
        
        # Check for execution results
        execution_steps = {k: v for k, v in result.get('results', {}).items() if 'execution' in k}
        print(f"🔍 Execution steps found: {list(execution_steps.keys())}")
        
        for step_name, step_data in execution_steps.items():
            print(f"\n📊 Step: {step_name}")
            print(f"  - Status: {step_data.get('status')}")
            print(f"  - Has results: {bool(step_data.get('results'))}")
            print(f"  - Results count: {len(step_data.get('results', []))}")
            if step_data.get('results'):
                print(f"  - First result: {step_data['results'][0] if step_data['results'] else 'None'}")
                
    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.text[:500])
        
except Exception as e:
    print(f"❌ Request failed: {e}")
