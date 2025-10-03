#!/usr/bin/env python3
"""
Test the retry logic with actual correct column names
"""

import sys
import os
from pathlib import Path
import requests
import json

def main():
    print("🎯 Testing with CORRECT column names based on reindexed schema")
    print("=" * 80)
    
    # Use the correct API endpoint to test retry logic
    base_url = "http://localhost:8000"
    
    # Build test data with corrected column names based on what we know exists
    test_data = {
        "query": "Show me territories with performance data",
        "user_id": "test_user",
        "conversation_context": {
            "user_id": "test_user", 
            "session_id": "test_session",
            "recent_queries": [],
            "is_follow_up": False,
            "follow_up_context": {},
            "available_tables": []
        }
    }
    
    print(f"📤 Sending test query: {test_data['query']}")
    
    try:
        # Use the agent query endpoint
        response = requests.post(f"{base_url}/api/agent/query", json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Query response received")
            
            # Check the SQL that was generated
            if 'sql' in result:
                print(f"🔍 Generated SQL:")
                print(result['sql'])
            
            # Check if execution succeeded
            if 'results' in result:
                results = result['results']
                if isinstance(results, dict) and results.get('success'):
                    print(f"✅ Query executed successfully!")
                    print(f"📊 Rows returned: {results.get('row_count', 0)}")
                else:
                    print(f"❌ Query execution failed")
                    if 'error_message' in results:
                        print(f"🚨 Error: {results['error_message']}")
            
            # Check if retry logic was used
            if 'retry_info' in result:
                print(f"🔄 Retry info: {result['retry_info']}")
            
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()