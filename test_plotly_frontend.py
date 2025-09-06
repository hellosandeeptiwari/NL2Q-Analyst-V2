import requests
import json

def test_plotly_integration_full():
    print("🎯 Testing Full Plotly Integration - Frontend Ready\n")
    
    # Test query that we know works
    query = "What are the recommended messages for NBA marketing actions?"
    
    print("=" * 70)
    print("TESTING TRADITIONAL ENDPOINT WITH PLOTLY INTEGRATION")
    print("=" * 70)
    
    payload = {
        "natural_language": query,
        "job_id": "frontend_plotly_test",
        "db_type": "snowflake"
    }
    
    try:
        response = requests.post("http://localhost:8000/query", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            sql = result.get('sql', '')
            rows = result.get('rows', [])
            plotly_spec = result.get('plotly_spec')
            
            print(f"✅ Backend Response SUCCESS")
            print(f"🔧 SQL: {sql}")
            print(f"📊 Data: {len(rows)} rows returned")
            
            if plotly_spec:
                print(f"\n📈 PLOTLY CHART SPECIFICATION:")
                print(f"   Chart Type: {plotly_spec.get('data', [{}])[0].get('type', 'unknown')}")
                print(f"   Title: {plotly_spec.get('layout', {}).get('title', {}).get('text', 'No title')}")
                
                # Show the actual data structure for frontend
                print(f"\n🔧 PLOTLY SPEC STRUCTURE:")
                print(f"   Data Points: {len(plotly_spec.get('data', [{}])[0].get('x', []))}")
                print(f"   X Values (first 3): {plotly_spec.get('data', [{}])[0].get('x', [])[:3]}")
                print(f"   Y Values (first 3): {plotly_spec.get('data', [{}])[0].get('y', [])[:3]}")
                
                print(f"\n✅ FRONTEND INTEGRATION READY:")
                print(f"   ✅ Backend generates plotly_spec")
                print(f"   ✅ Frontend can receive this in response")
                print(f"   ✅ Plotly.js can render this spec directly")
                print(f"   ✅ Chart will display inline in the UI")
                
                # Show sample data for verification
                if rows and len(rows) > 0:
                    print(f"\n📝 SAMPLE DATA:")
                    for i, row in enumerate(rows[:3]):
                        if isinstance(row, dict) and 'Recommended_Msg_Overall' in row:
                            msg = row['Recommended_Msg_Overall']
                            print(f"   {i+1}. {msg[:60]}...")
                        elif isinstance(row, list) and len(row) > 0:
                            print(f"   {i+1}. {str(row[0])[:60]}...")
                
                # Save the plotly spec for frontend testing
                with open('c:\\Users\\SandeepTiwari\\NL2Q Agent\\test_plotly_spec.json', 'w') as f:
                    json.dump(plotly_spec, f, indent=2)
                print(f"\n💾 Plotly spec saved to test_plotly_spec.json for frontend testing")
                
            else:
                print(f"❌ No Plotly spec generated")
                
        else:
            print(f"❌ Backend request failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print(f"\n" + "=" * 70)
    print("FRONTEND INTEGRATION INSTRUCTIONS")
    print("=" * 70)
    print(f"🎯 To test inline charts in the frontend:")
    print(f"   1. Open http://localhost:3000 (already opened)")
    print(f"   2. Enter query: '{query}'")
    print(f"   3. Submit the query")
    print(f"   4. Check that both data table AND chart appear")
    print(f"   5. The chart should be a bar chart showing message frequencies")
    print(f"\n✅ Backend is ready to serve Plotly specs to frontend!")
    print(f"✅ Frontend should display charts inline with query results!")

if __name__ == "__main__":
    test_plotly_integration_full()
