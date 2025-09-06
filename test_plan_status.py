import requests
import json

def test_plan_status_endpoint():
    print("🔍 Testing plan status endpoint...\n")
    
    # First, create a query to get a plan_id
    query_payload = {
        "query": "read table final nba output python and fetch top 5 rows",
        "user_id": "test_user",
        "session_id": "test_session"
    }
    
    try:
        # Submit query
        print("1. Submitting query to get plan_id...")
        response = requests.post("http://localhost:8000/api/agent/query", json=query_payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            plan_id = result.get('plan_id')
            print(f"✅ Got plan_id: {plan_id}")
            
            if plan_id:
                # Test plan status endpoint
                print(f"\n2. Testing plan status endpoint...")
                status_response = requests.get(f"http://localhost:8000/api/agent/plan/{plan_id}/status", timeout=10)
                
                if status_response.status_code == 200:
                    status_result = status_response.json()
                    print(f"✅ Plan status endpoint works!")
                    print(f"📊 Status: {status_result.get('status', 'unknown')}")
                    print(f"📊 Progress: {status_result.get('progress', 0)}%")
                    print(f"📊 Current step: {status_result.get('current_step', 'unknown')}")
                    
                    # Check for results and charts
                    if 'results' in status_result:
                        results = status_result['results']
                        print(f"📊 Results keys: {list(results.keys())}")
                        
                        for key, value in results.items():
                            if isinstance(value, dict) and 'charts' in value:
                                charts = value['charts']
                                print(f"🎨 Found {len(charts)} charts in {key}")
                                for i, chart in enumerate(charts):
                                    print(f"   Chart {i+1}: {chart.get('type', 'unknown')} - {chart.get('title', 'no title')}")
                else:
                    print(f"❌ Plan status endpoint failed: {status_response.status_code}")
                    print(f"Response: {status_response.text}")
            else:
                print(f"❌ No plan_id in response")
        else:
            print(f"❌ Query submission failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_plan_status_endpoint()
