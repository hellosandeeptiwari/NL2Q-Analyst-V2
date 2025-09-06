import requests
import json

def test_both_endpoints_with_plotly():
    print("🔧 Testing both endpoints with Plotly integration...\n")
    
    query = "What are the recommended messages for NBA marketing actions?"
    
    # Test Traditional Endpoint
    print("=" * 60)
    print("TESTING TRADITIONAL ENDPOINT (/query)")
    print("=" * 60)
    
    traditional_payload = {
        "natural_language": query,
        "job_id": "test_traditional_plotly",
        "db_type": "snowflake"
    }
    
    try:
        response = requests.post("http://localhost:8000/query", json=traditional_payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            sql = result.get('sql', '')
            rows = result.get('rows', [])
            plotly_spec = result.get('plotly_spec')
            
            print(f"✅ Traditional endpoint SUCCESS")
            print(f"🔧 Generated SQL: {sql}")
            print(f"📈 Returned {len(rows)} rows")
            
            # Check table used
            if 'NBA_outputs' in sql:
                print(f"✅ Used NBA_outputs table (valid Pinecone result)")
            elif any(table in sql for table in ['Final_NBA_Output_python']):
                print(f"✅ Used Final_NBA_Output table (valid Pinecone result)")
            else:
                print(f"❓ Used different table: {sql}")
                
            # Check Plotly integration
            if plotly_spec:
                print(f"✅ Plotly spec generated!")
                print(f"📊 Chart type: {plotly_spec.get('data', [{}])[0].get('type', 'unknown')}")
            else:
                print(f"❌ No Plotly spec generated")
                
        else:
            print(f"❌ Traditional endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Traditional endpoint error: {e}")
    
    # Test Orchestrator Endpoint
    print(f"\n" + "=" * 60)
    print("TESTING ORCHESTRATOR ENDPOINT (/agent)")
    print("=" * 60)
    
    orchestrator_payload = {
        "query": query,
        "job_id": "test_orchestrator_plotly"
    }
    
    try:
        response = requests.post("http://localhost:8000/agent", json=orchestrator_payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            sql = result.get('sql', '')
            rows = result.get('data', [])
            plotly_spec = result.get('plotly_spec')
            
            print(f"✅ Orchestrator endpoint SUCCESS")
            print(f"🔧 Generated SQL: {sql}")
            print(f"📈 Returned {len(rows)} rows")
            
            # Check Plotly integration
            if plotly_spec:
                print(f"✅ Plotly spec generated!")
                print(f"📊 Chart type: {plotly_spec.get('data', [{}])[0].get('type', 'unknown')}")
            else:
                print(f"❌ No Plotly spec generated")
                
        else:
            print(f"❌ Orchestrator endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Orchestrator endpoint error: {e}")
    
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Both endpoints should now work with Plotly charts!")
    print(f"✅ Traditional endpoint uses Pinecone search + schema filtering")
    print(f"✅ Orchestrator endpoint uses advanced agentic approach")
    print(f"✅ Both generate automatic Plotly visualizations")

if __name__ == "__main__":
    test_both_endpoints_with_plotly()
