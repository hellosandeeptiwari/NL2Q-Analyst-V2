#!/usr/bin/env python3
"""
Test the actual backend server with our improved dynamic SQL generation
"""

import requests
import json
import time

def test_server_queries():
    """Test real server endpoints with various queries"""
    server_url = "http://localhost:8000"
    
    print("🚀 Testing Real Backend Server")
    print("=" * 80)
    
    # Test queries
    test_queries = [
        "summarize top 10 sales of tirosint sol by territory",
        "show me top 5 prescriptions of Levothyroxine by region", 
        "get top 15 records by sales volume",
        "display all prescriber data",
        "find data for product Synthroid in territories"
    ]
    
    print("🔗 Checking server availability...")
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
        else:
            print(f"⚠️ Server responded with status {response.status_code}")
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        print("💡 Make sure to start the server with: python -m uvicorn backend.main:app --reload")
        return
    
    print("\n🎯 Testing Query Processing...")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}: {query}")
        print("-" * 40)
        
        try:
            # Test the /query endpoint (fix the parameter name issue)
            payload = {"query": query}  # Try "query" first
            response = requests.post(f"{server_url}/query", json=payload, timeout=30)
            
            if response.status_code != 200:
                # Try with "natural_language" parameter
                payload = {"natural_language": query}
                response = requests.post(f"{server_url}/query", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Status: {response.status_code}")
                
                # Analyze the result
                if 'sql_query' in result:
                    sql = result['sql_query']
                    print(f"📊 Generated SQL:")
                    print(sql)
                    
                    # Check for our improvements
                    sql_upper = sql.upper()
                    improvements = []
                    
                    if 'TOP ' in sql_upper:
                        top_match = None
                        for num in ['100', '15', '10', '5']:
                            if f'TOP {num}' in sql_upper:
                                top_match = num
                                break
                        if top_match:
                            improvements.append(f"✅ Dynamic TOP {top_match}")
                    
                    if 'WHERE' in sql_upper:
                        improvements.append("✅ Smart filtering")
                    
                    if 'ORDER BY' in sql_upper:
                        improvements.append("✅ Intelligent ordering")
                    
                    if 'TIROSINT' in sql_upper or 'LEVOTHYROXINE' in sql_upper or 'SYNTHROID' in sql_upper:
                        improvements.append("✅ Product detection")
                    
                    if improvements:
                        print(f"🎯 Detected Improvements: {', '.join(improvements)}")
                    else:
                        print("⚠️ No specific improvements detected")
                
                if 'error' in result:
                    print(f"⚠️ Query had errors: {result['error']}")
                
                if 'execution_time' in result:
                    print(f"⏱️ Execution time: {result['execution_time']}s")
                    
            else:
                print(f"❌ Request failed: {response.status_code}")
                if response.text:
                    print(f"Error: {response.text}")
                    
        except Exception as e:
            print(f"❌ Error testing query: {e}")
        
        time.sleep(1)  # Brief pause between requests
    
    print("\n" + "=" * 80)
    print("🏁 Server Testing Complete")
    print("💡 Check the results above to verify dynamic SQL generation is working!")

if __name__ == "__main__":
    test_server_queries()