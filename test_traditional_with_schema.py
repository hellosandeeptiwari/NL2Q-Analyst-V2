import requests
import json

def test_traditional_endpoint_with_schema():
    print("🔍 Testing traditional endpoint with populated schema cache...\n")
    
    query = "What are the recommended messages for NBA marketing actions?"
    
    payload = {
        "natural_language": query,
        "job_id": "test_with_schema",
        "db_type": "snowflake"
    }
    
    try:
        response = requests.post("http://localhost:8000/query", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ SUCCESS! Traditional endpoint working with populated schema")
            print(f"📊 Response keys: {list(result.keys())}")
            
            # Check SQL
            sql = result.get('sql', 'No SQL found')
            print(f"\n🔧 Generated SQL:")
            print("-" * 60)
            print(sql)
            print("-" * 60)
            
            # Analyze the SQL for correct column usage
            if "Recommended_Msg_Overall" in sql:
                print(f"🎯 PERFECT: SQL contains correct column 'Recommended_Msg_Overall'!")
            elif "Final_NBA_Output" in sql:
                print(f"🎯 GOOD: SQL targets correct NBA table family")
                if "Recommended_Msg_Overall" not in sql:
                    print(f"⚠️ PARTIAL: Correct table but missing target column")
            elif "recommended_message" in sql.lower():
                print(f"❌ STILL BROKEN: SQL contains invalid column 'recommended_message'")
            else:
                print(f"❓ UNCLEAR: SQL doesn't contain expected patterns")
            
            # Check execution results
            rows = result.get('rows', [])
            print(f"\n📈 Query returned {len(rows)} rows")
            
            if rows:
                print(f"🔍 Sample columns: {list(rows[0].keys()) if isinstance(rows[0], dict) else 'Raw data'}")
                print(f"🔍 First few rows:")
                for i, row in enumerate(rows[:3]):
                    print(f"   Row {i+1}: {row}")
                    
                # Check if we got the target column in results
                if rows and isinstance(rows[0], dict):
                    if 'Recommended_Msg_Overall' in rows[0]:
                        print(f"🎯 PERFECT: Results contain target column 'Recommended_Msg_Overall'!")
                    else:
                        print(f"📋 Available result columns: {list(rows[0].keys())}")
                
                # Check Plotly specification
                plotly_spec = result.get('plotly_spec', {})
                if plotly_spec:
                    print(f"\n📊 PLOTLY SPEC GENERATED!")
                    chart_data = plotly_spec.get('data', [])
                    if chart_data:
                        chart_type = chart_data[0].get('type', 'unknown')
                        title = plotly_spec.get('layout', {}).get('title', {}).get('text', 'No title')
                        x_data = chart_data[0].get('x', [])
                        y_data = chart_data[0].get('y', [])
                        
                        print(f"   📊 Chart type: {chart_type}")
                        print(f"   📊 Title: {title}")
                        print(f"   📊 Data points: {len(x_data)}")
                        print(f"   📊 Sample X data: {x_data[:3] if x_data else 'None'}")
                        print(f"   📊 Sample Y data: {y_data[:3] if y_data else 'None'}")
                else:
                    print(f"\n⚠️ No Plotly spec generated (expected since no data returned)")
            else:
                print(f"📈 No data rows returned - likely table doesn't exist or query failed")
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_traditional_endpoint_with_schema()
