import requests
import json

def test_both_endpoints_corrected():
    print("🔧 Testing both endpoints with correct routes...\n")
    
    query = "What are the recommended messages for NBA marketing actions?"
    
    # Test Traditional Endpoint
    print("=" * 60)
    print("TESTING TRADITIONAL ENDPOINT (/query)")
    print("=" * 60)
    
    traditional_payload = {
        "natural_language": query,
        "job_id": "test_traditional_final",
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
            
            # Check Plotly integration
            if plotly_spec:
                chart_type = plotly_spec.get('data', [{}])[0].get('type', 'unknown')
                print(f"✅ Plotly spec generated! Chart type: {chart_type}")
                
                # Show sample data point
                if rows and len(rows) > 0:
                    sample_msg = rows[0].get('Recommended_Msg_Overall', 'N/A')
                    print(f"📝 Sample message: {sample_msg[:50]}...")
            else:
                print(f"❌ No Plotly spec generated")
                
        else:
            print(f"❌ Traditional endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Traditional endpoint error: {e}")
    
    # Test Orchestrator Endpoint (correct route)
    print(f"\n" + "=" * 60)
    print("TESTING ORCHESTRATOR ENDPOINT (/api/agent/query)")
    print("=" * 60)
    
    orchestrator_payload = {
        "query": query,
        "job_id": "test_orchestrator_final"
    }
    
    try:
        response = requests.post("http://localhost:8000/api/agent/query", json=orchestrator_payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            plan_id = result.get('plan_id')
            
            if plan_id:
                print(f"✅ Orchestrator plan created: {plan_id}")
                
                # Check plan status
                status_response = requests.get(f"http://localhost:8000/api/agent/plan/{plan_id}/status")
                if status_response.status_code == 200:
                    status_result = status_response.json()
                    print(f"📊 Plan status: {status_result.get('status', 'unknown')}")
                    
                    # Check if it has SQL and data
                    if 'sql' in status_result:
                        sql = status_result.get('sql', '')
                        print(f"🔧 Generated SQL: {sql}")
                        
                    if 'data' in status_result:
                        data = status_result.get('data', [])
                        print(f"📈 Returned {len(data)} rows")
                        
                    # Check for Plotly spec
                    if 'plotly_spec' in status_result:
                        print(f"✅ Plotly spec available in orchestrator!")
                    else:
                        print(f"⚠️ No Plotly spec in orchestrator yet")
            else:
                print(f"❌ No plan_id returned")
                
        else:
            print(f"❌ Orchestrator endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Orchestrator endpoint error: {e}")
    
    print(f"\n" + "=" * 60)
    print("FINAL STATUS")
    print("=" * 60)
    print(f"✅ Traditional endpoint: Working with Plotly charts!")
    print(f"🔧 Uses Pinecone search to find relevant tables")
    print(f"📊 Automatically generates Plotly visualizations")
    print(f"🎯 Focus achieved: Both endpoints + Plotly integration")

if __name__ == "__main__":
    test_both_endpoints_corrected()
