"""Test query to see comprehensive debug output"""
import requests
import json

url = "http://localhost:8000/api/agent/query"
payload = {
    "query": "Compare TRX vs NRX share performance for Flector, Licart, and Tirosint across all regions",
    "user_id": "test_debug"
}

print("🚀 Sending query to backend...")
print(f"Query: {payload['query']}")
print("-" * 80)

try:
    response = requests.post(url, json=payload, timeout=120)
    
    print(f"\n📥 Response Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        # Check for intelligent visualization planning in context
        if 'context' in data:
            print(f"\n✅ Context found in response")
            print(f"Context keys: {list(data['context'].keys())}")
            
            if 'intelligent_visualization_planning' in data['context']:
                print(f"\n✅✅✅ INTELLIGENT VISUALIZATION PLANNING FOUND! ✅✅✅")
                ivp = data['context']['intelligent_visualization_planning']
                print(f"Status: {ivp.get('status', 'unknown')}")
                
                if 'visualization_plan' in ivp:
                    vp = ivp['visualization_plan']
                    print(f"\nVisualization Plan:")
                    print(f"  Layout: {vp.get('layout_type', 'unknown')}")
                    print(f"  KPIs: {len(vp.get('kpis', []))}")
                    print(f"  Charts: {len(vp.get('charts', []))}")
                    
                    if vp.get('kpis'):
                        print(f"\n  KPI Details:")
                        for i, kpi in enumerate(vp['kpis'][:3], 1):
                            print(f"    {i}. {kpi.get('title', 'Unknown')}")
                    
                    if vp.get('charts'):
                        print(f"\n  Chart Details:")
                        for i, chart in enumerate(vp['charts'][:3], 1):
                            print(f"    {i}. Type: {chart.get('type', 'unknown')}, Title: {chart.get('title', 'Unknown')}")
                else:
                    print(f"❌ No visualization_plan in intelligent_visualization_planning")
            else:
                print(f"\n❌ intelligent_visualization_planning NOT FOUND in context")
                print(f"Available context keys: {list(data['context'].keys())}")
        else:
            print(f"\n❌ No context in response")
            print(f"Response keys: {list(data.keys())}")
        
        # Save response for inspection
        with open('last_api_response.json', 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n💾 Full response saved to last_api_response.json")
        
    else:
        print(f"❌ Error: {response.text}")
        
except Exception as e:
    print(f"❌ Exception: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "-" * 80)
print("✅ Test complete. Check terminal running uvicorn for debug logs!")
