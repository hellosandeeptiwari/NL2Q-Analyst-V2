#!/usr/bin/env python3
"""
Test script for GA providers query followed by line chart conversion
"""

import asyncio
import requests
import time
import json

BASE_URL = "http://localhost:8000"

async def test_ga_providers_followup():
    """Test GA providers query followed by line chart conversion"""
    
    print("🔍 Testing GA providers query with line chart follow-up...")
    
    # Step 1: First query about GA providers
    first_query = "for GA state, which providers are most expensive relative to average?"
    print(f"\n📊 Step 1: Running initial query: '{first_query}'")
    
    response1 = requests.post(
        f"{BASE_URL}/api/agent/query",
        json={
            "query": first_query,
            "user_id": "test_user"
        },
        timeout=30
    )
    
    if response1.status_code != 200:
        print(f"❌ First query failed: {response1.status_code}")
        print(f"Response: {response1.text}")
        return
    
    result1 = response1.json()
    print(f"✅ First query completed successfully")
    print(f"📊 Got {len(result1.get('rows', []))} rows")
    
    if result1.get('rows'):
        print(f"📋 Sample data: {result1['rows'][:3]}")
    
    # Give a small delay to ensure data is saved
    time.sleep(1)
    
    # Step 2: Follow-up query for line chart
    followup_query = "convert this into line chart"
    print(f"\n📈 Step 2: Running follow-up query: '{followup_query}'")
    
    response2 = requests.post(
        f"{BASE_URL}/api/agent/query", 
        json={
            "query": followup_query,
            "user_id": "test_user"  # Same user to ensure context
        },
        timeout=30
    )
    
    if response2.status_code != 200:
        print(f"❌ Follow-up query failed: {response2.status_code}")
        print(f"Response: {response2.text}")
        return
    
    result2 = response2.json()
    print(f"✅ Follow-up query completed")
    
    # Check if we got sample data (indicates the bug)
    if result2.get('python_code'):
        code = result2['python_code']
        if "Sample A" in code or "Sample B" in code:
            print("❌ BUG DETECTED: Follow-up query used sample data instead of real data!")
            print("🔍 Sample data indicators found in generated code")
        else:
            print("✅ SUCCESS: Follow-up query appears to use real data")
    
    print(f"\n📊 Follow-up result preview:")
    print(f"   Has python_code: {bool(result2.get('python_code'))}")
    print(f"   Code length: {len(result2.get('python_code', ''))}")
    
    if result2.get('python_code'):
        # Show first few lines to see data source
        lines = result2['python_code'].split('\n')[:10]
        print("🔍 First 10 lines of generated code:")
        for i, line in enumerate(lines, 1):
            print(f"   {i:2d}: {line}")

def main():
    """Main test function"""
    try:
        # Test server connectivity first
        response = requests.get(f"{BASE_URL}/api/database/indexing-status", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"⚠️ Server responded with status: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Make sure the server is running on http://localhost:8000")
        return
    
    # Run the test
    asyncio.run(test_ga_providers_followup())

if __name__ == "__main__":
    main()
