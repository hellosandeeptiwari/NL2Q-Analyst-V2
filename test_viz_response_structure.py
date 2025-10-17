"""
Test script to verify intelligent visualization response structure
"""
import requests
import json

# Test the API response structure
def test_viz_response():
    url = "http://localhost:8000/api/agent/query"
    
    payload = {
        "query": "Show me top 5 prescribers by TRX for Tirosint",
        "user_id": "test_user",
        "session_id": "test_session",
        "use_deterministic": True
    }
    
    print("🧪 Testing visualization response structure...")
    print(f"📤 Sending request: {payload['query']}")
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"\n✅ Response received (Status: {response.status_code})")
        print(f"📊 Response keys: {list(data.keys())}")
        
        # Check for context
        if 'context' in data:
            print(f"\n✅ 'context' key found!")
            print(f"📊 Context keys: {list(data['context'].keys())}")
            
            # Check for intelligent_visualization_planning
            if 'intelligent_visualization_planning' in data['context']:
                print(f"\n🎉 'intelligent_visualization_planning' FOUND!")
                viz_plan = data['context']['intelligent_visualization_planning']
                print(f"📊 Viz plan keys: {list(viz_plan.keys())}")
                
                if 'visualization_plan' in viz_plan:
                    print(f"\n✨ Visualization Plan Details:")
                    plan = viz_plan['visualization_plan']
                    print(json.dumps(plan, indent=2))
                    
                    print(f"\n✅ SUCCESS! Frontend should display intelligent visualization")
                    print(f"   Layout: {plan.get('layout', 'Unknown')}")
                    print(f"   KPIs: {len(plan.get('kpis', []))}")
                    print(f"   Charts: {len(plan.get('charts', []))}")
                else:
                    print(f"\n❌ 'visualization_plan' not found in intelligent_visualization_planning")
            else:
                print(f"\n❌ 'intelligent_visualization_planning' NOT found in context")
                print(f"   Available context keys: {list(data['context'].keys())}")
        else:
            print(f"\n❌ 'context' key NOT found in response")
            print(f"   Available top-level keys: {list(data.keys())}")
            
            # Check if results contains the viz plan
            if 'results' in data:
                print(f"\n🔍 Checking 'results' for viz plan...")
                results = data['results']
                print(f"   Results keys: {list(results.keys())}")
                
                for key, value in results.items():
                    if 'viz' in key.lower():
                        print(f"   Found viz-related key: {key}")
    else:
        print(f"\n❌ Request failed (Status: {response.status_code})")
        print(f"   Response: {response.text}")

if __name__ == "__main__":
    test_viz_response()
