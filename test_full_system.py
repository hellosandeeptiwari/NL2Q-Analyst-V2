#!/usr/bin/env python3

import asyncio
import requests
import json

async def test_full_system():
    """Test the complete system end-to-end"""
    
    print("🔍 Testing complete system workflow...")
    
    # Test 1: Check if backend is running
    try:
        response = requests.get("http://localhost:8000/api/database/status", timeout=5)
        print(f"✅ Backend is running: {response.status_code}")
        print(f"Database status: {response.json()}")
    except Exception as e:
        print(f"❌ Backend not accessible: {e}")
        return
    
    # Test 2: Send a query to the agent
    query_data = {
        "query": "show me top 5 rows from final nba output table",
        "user_id": "default_user"
    }
    
    try:
        print(f"🚀 Sending query: {query_data['query']}")
        response = requests.post(
            "http://localhost:8000/api/agent/query",
            json=query_data,
            timeout=60
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Query successful!")
            print(f"Plan status: {result.get('plan_status', 'unknown')}")
            print(f"Results keys: {list(result.keys())}")
            
            # Check for SQL execution results
            if 'results' in result:
                sql_results = result['results'].get('3_sql_execution', {})
                if sql_results:
                    print(f"SQL execution success: {sql_results.get('success', False)}")
                    if sql_results.get('success'):
                        print(f"Data rows: {len(sql_results.get('data', []))}")
                    else:
                        print(f"SQL error: {sql_results.get('error', 'Unknown error')}")
            
            # Check for visualization results  
            if 'results' in result:
                viz_results = result['results'].get('7_visualization', {})
                if viz_results:
                    print(f"Visualization success: {viz_results.get('success', False)}")
                    if viz_results.get('success'):
                        charts = viz_results.get('charts', [])
                        print(f"Charts generated: {len(charts)}")
                    else:
                        print(f"Visualization error: {viz_results.get('error', 'Unknown error')}")
                        
        else:
            print(f"❌ Query failed with status {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Query request failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_full_system())
