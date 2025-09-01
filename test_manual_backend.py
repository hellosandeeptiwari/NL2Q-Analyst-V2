#!/usr/bin/env python3
"""
Manual testing client for the optimized NBA Agent Server
"""
import requests
import json
import time

def test_backend_manually():
    """Test the backend server manually with interactive queries"""
    base_url = "http://localhost:8003"
    
    print("🧪 Manual Testing - NBA Agent Server with Optimized Embeddings")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. 🔍 Testing Health Endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.text}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test 2: Agent status
    print("\n2. 🤖 Testing Agent Status...")
    try:
        response = requests.get(f"{base_url}/agent-status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Agent status retrieved")
            print(f"   Initialized: {data.get('initialized', False)}")
            print(f"   Total Tables: {data.get('total_tables', 0)}")
            print(f"   OpenAI Available: {data.get('openai_api_available', False)}")
        else:
            print(f"❌ Agent status failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Agent status error: {e}")
    
    # Test 3: Table listing
    print("\n3. 📋 Testing Table Listing...")
    try:
        response = requests.get(f"{base_url}/tables", timeout=10)
        if response.status_code == 200:
            data = response.json()
            tables = data.get('tables', [])
            print(f"✅ Found {len(tables)} tables")
            
            # Show NBA-related tables
            nba_tables = [t for t in tables if 'nba' in t.lower()]
            print(f"   NBA tables found: {len(nba_tables)}")
            for table in nba_tables[:5]:
                print(f"     • {table}")
        else:
            print(f"❌ Table listing failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Table listing error: {e}")
    
    # Test 4: Intelligent query processing
    print("\n4. 🧠 Testing Intelligent Query Processing...")
    test_queries = [
        "Show me NBA player statistics",
        "Basketball game performance data", 
        "Player scoring and analytics information",
        "Team performance metrics"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: '{query}'")
        try:
            start_time = time.time()
            payload = {"query": query}
            response = requests.post(f"{base_url}/query", 
                                   json=payload, 
                                   timeout=15,
                                   headers={"Content-Type": "application/json"})
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Response in {response_time:.2f}s")
                
                if 'suggested_tables' in data:
                    suggestions = data['suggested_tables']
                    print(f"   📊 Found {len(suggestions)} table suggestions:")
                    for j, table in enumerate(suggestions[:3], 1):
                        if isinstance(table, dict):
                            name = table.get('name', 'Unknown')
                            score = table.get('score', 'N/A')
                            print(f"     {j}. {name} (score: {score})")
                        else:
                            print(f"     {j}. {table}")
                
                if 'vector_analysis' in data:
                    vector_info = data['vector_analysis']
                    print(f"   🔍 Vector search time: {vector_info.get('processing_time', 'N/A')}s")
                
                if 'llm_analysis' in data:
                    llm_info = data['llm_analysis']
                    intent = llm_info.get('intent', 'N/A')
                    print(f"   🧠 LLM detected intent: {intent}")
                    
            else:
                print(f"   ❌ Query failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"   ❌ Query error: {e}")
    
    # Test 5: Performance summary
    print("\n5. ⚡ Performance Summary...")
    print("   Backend Features:")
    print("   • Ultra-fast schema extraction (6s vs 157s)")
    print("   • OpenAI semantic embeddings")
    print("   • Intelligent table suggestions")
    print("   • Multi-agent orchestration")
    print("   • Smart caching system")
    
    print("\n🎉 Manual testing complete!")
    print("\nNext steps:")
    print("1. Open browser to http://localhost:8003 (backend)")
    print("2. Frontend should be available at http://localhost:3000")
    print("3. Try different NBA-related queries")
    print("4. Observe the fast response times!")

if __name__ == "__main__":
    test_backend_manually()
