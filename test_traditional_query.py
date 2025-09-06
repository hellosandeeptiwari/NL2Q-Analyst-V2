import requests
import json

def test_traditional_query_endpoint():
    print("🔍 Testing traditional /query endpoint with populated schema cache...\n")
    
    # Test the query that was failing before
    query = "What are the recommended messages for NBA marketing actions?"
    
    url = "http://localhost:8000/query"
    payload = {
        "natural_language": query,  # Note: different key than orchestrator
        "job_id": "test_traditional_query",
        "db_type": "snowflake"
    }
    
    print(f"📡 Sending query: {query}")
    print(f"🔗 URL: {url}")
    print(f"📦 Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        
        print(f"🌐 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ SUCCESS! Traditional query endpoint working")
            print(f"📊 Response keys: {list(result.keys())}")
            
            # Check SQL
            sql = result.get('sql', 'No SQL found')
            print(f"\n🔧 Generated SQL:")
            print("-" * 50)
            print(sql)
            print("-" * 50)
            
            # Analyze the SQL for correct column usage
            if "Recommended_Msg_Overall" in sql:
                print(f"✅ PERFECT: SQL contains correct column 'Recommended_Msg_Overall'!")
            elif "recommended_message" in sql.lower():
                print(f"❌ STILL BROKEN: SQL contains invalid column 'recommended_message'")
            elif "SELECT NULL" in sql or "information_schema" in sql:
                print(f"⚠️ FALLBACK: Using fallback SQL - schema cache might not be loaded")
            else:
                print(f"❓ UNCLEAR: SQL doesn't contain expected patterns")
            
            # Check execution results
            rows = result.get('rows', [])
            if rows:
                print(f"\n📈 Query returned {len(rows)} rows")
                if rows:
                    print(f"🔍 Sample columns: {list(rows[0].keys()) if isinstance(rows[0], dict) else 'Raw data'}")
                    print(f"🔍 First few rows:")
                    for i, row in enumerate(rows[:3]):
                        print(f"   Row {i+1}: {row}")
            else:
                print(f"\n📈 No data rows returned")
                
            # Check Plotly specification
            plotly_spec = result.get('plotly_spec', {})
            if plotly_spec:
                print(f"\n📊 Plotly spec available: {bool(plotly_spec)}")
                print(f"📊 Plotly keys: {list(plotly_spec.keys())}")
            else:
                print(f"\n📊 No Plotly specification provided")
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_traditional_query_endpoint()
