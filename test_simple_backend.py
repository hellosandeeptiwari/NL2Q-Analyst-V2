#!/usr/bin/env python3
"""
Simple test to check backend endpoint parameter handling
"""

import requests
import json

def test_backend_endpoint():
    """Test the backend endpoint directly"""
    print("🔍 Testing Backend Endpoint Parameter Handling")
    print("=" * 60)
    
    backend_url = "http://localhost:8000"
    test_query = "summarize top 10 sales of tirosint sol by territory"
    
    # Test different parameter combinations
    test_payloads = [
        {"query": test_query},
        {"nl": test_query}, 
        {"natural_language": test_query},
        {"query": test_query, "job_id": "test123"}
    ]
    
    for i, payload in enumerate(test_payloads, 1):
        print(f"\n📝 Test {i}: {list(payload.keys())}")
        print("-" * 40)
        
        try:
            response = requests.post(
                f"{backend_url}/query",
                json=payload,
                timeout=10
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Success!")
                if 'sql_query' in result:
                    sql = result['sql_query'][:100] + "..." if len(result['sql_query']) > 100 else result['sql_query']
                    print(f"SQL: {sql}")
                if 'error' in result:
                    print(f"Error: {result['error']}")
                break
            else:
                print(f"❌ Failed: {response.text[:200]}")
                
        except requests.exceptions.Timeout:
            print("❌ Request timed out")
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to backend")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n💡 Check the backend terminal for debug output!")

if __name__ == "__main__":
    test_backend_endpoint()