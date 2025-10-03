#!/usr/bin/env python3
"""
Test script to analyze backend query generation and identify fallback usage
"""

import requests
import json
import time

def test_backend_query_generation():
    """Test backend and analyze why it's using fallback instead of main query generation"""
    print("🔍 Backend Query Generation Analysis")
    print("=" * 80)
    
    backend_url = "http://localhost:8000"
    
    # Test queries to analyze
    test_queries = [
        "summarize top 10 sales of tirosint sol by territory",
        "show me top 5 prescriptions by region",
        "get all prescriber data",
        "find data for levothyroxine products"
    ]
    
    print("🔗 Checking backend availability...")
    try:
        response = requests.get(f"{backend_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend server is running!")
        else:
            print(f"⚠️ Backend responded with status {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Backend not accessible: {e}")
        print("💡 Make sure backend is running: python -m uvicorn backend.main:app --reload")
        return
    
    print("\n🎯 Testing Query Generation and Analyzing Fallback Usage...")
    print("-" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}: {query}")
        print("-" * 60)
        
        try:
            # Test with different parameter names
            payloads = [
                {"query": query},
                {"natural_language": query},
                {"nl": query}
            ]
            
            response = None
            for payload in payloads:
                try:
                    response = requests.post(
                        f"{backend_url}/query", 
                        json=payload, 
                        timeout=30
                    )
                    if response.status_code == 200:
                        print(f"✅ Successful with payload: {list(payload.keys())[0]}")
                        break
                except:
                    continue
            
            if not response or response.status_code != 200:
                print(f"❌ All requests failed")
                continue
                
            result = response.json()
            
            # Analyze the response for fallback indicators
            print(f"📊 Response Analysis:")
            
            if 'sql_query' in result:
                sql = result['sql_query']
                print(f"Generated SQL:")
                print(sql[:200] + "..." if len(sql) > 200 else sql)
                
                # Check for fallback indicators
                fallback_indicators = []
                
                if "-- Dynamic template query" in sql:
                    fallback_indicators.append("🔴 Using template fallback")
                elif "-- Fallback query" in sql:
                    fallback_indicators.append("🔴 Using fallback generation")
                elif sql.strip().startswith("SELECT TOP"):
                    fallback_indicators.append("🟡 Possibly using template (starts with SELECT TOP)")
                else:
                    fallback_indicators.append("🟢 Using main LLM generation")
                
                if "WHERE" in sql.upper():
                    fallback_indicators.append("✅ Has filtering")
                if "ORDER BY" in sql.upper():
                    fallback_indicators.append("✅ Has ordering")
                if "TOP " in sql.upper():
                    fallback_indicators.append("✅ Has TOP clause")
                
                print(f"🔍 Analysis: {', '.join(fallback_indicators)}")
            
            if 'error' in result:
                print(f"⚠️ Query had errors: {result['error']}")
                if 'fallback' in str(result['error']).lower():
                    print("🔴 ERROR: Fallback triggered due to error")
            
            if 'execution_time' in result:
                print(f"⏱️ Execution time: {result['execution_time']}s")
            
            # Look for debug information
            if 'debug_info' in result:
                print(f"🐛 Debug info available")
            
            if 'query_plan' in result:
                print(f"📋 Query plan generated")
                
        except Exception as e:
            print(f"❌ Error testing query: {e}")
        
        time.sleep(0.5)  # Brief pause between requests
    
    print("\n" + "=" * 80)
    print("🔍 FALLBACK ANALYSIS SUMMARY")
    print("=" * 80)
    print("Common reasons for fallback usage:")
    print("1. 🚫 LLM API errors or timeouts")
    print("2. 🔌 Missing OpenAI API key or configuration")
    print("3. 🎯 Query complexity exceeding LLM capabilities")
    print("4. 📊 Schema information not properly loaded")
    print("5. 🔄 Template fallback triggered by safety mechanisms")
    print("\n💡 Check backend logs for specific error messages!")

if __name__ == "__main__":
    test_backend_query_generation()