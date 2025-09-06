import requests
import json

def test_with_longer_timeout():
    print("🔍 Testing with longer timeout...\n")
    
    query_payload = {
        "query": "read table final nba output python and fetch top 5 rows",
        "user_id": "test_user", 
        "session_id": "test_session"
    }
    
    try:
        print("1. Submitting query with 60 second timeout...")
        response = requests.post("http://localhost:8000/api/agent/query", json=query_payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            plan_id = result.get('plan_id')
            print(f"✅ Got plan_id: {plan_id}")
            print(f"📊 Status: {result.get('status', 'unknown')}")
            
            return plan_id
        else:
            print(f"❌ Query failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    test_with_longer_timeout()
