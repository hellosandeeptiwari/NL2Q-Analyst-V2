import requests
import json

def test_nba_query_with_fixed_schema():
    print("Testing NBA query with fixed schema extraction...\n")
    
    # Test the query that was failing before
    query = "What are the recommended messages for NBA marketing actions?"
    
    url = "http://localhost:8000/query"
    payload = {
        "question": query,
        "user_id": "test_user",
        "session_id": "test_session"
    }
    
    print(f"🔍 Sending query: {query}")
    print(f"📡 URL: {url}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        print(f"🌐 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ SUCCESS! Query completed")
            print(f"📊 Response keys: {list(result.keys())}")
            
            # Check if SQL was generated (try different possible keys)
            sql = None
            if 'sql_query' in result:
                sql = result['sql_query']
            elif 'sql' in result:
                sql = result['sql']
            elif 'query' in result:
                sql = result['query']
            
            if sql:
                print(f"🔧 Generated SQL:")
                print(sql)
                print()  # Add blank line for readability
                
                # Check for the correct column name
                if "Recommended_Msg_Overall" in sql:
                    print(f"✅ FIXED: SQL contains correct column 'Recommended_Msg_Overall'!")
                elif "recommended_message" in sql.lower():
                    print(f"❌ STILL BROKEN: SQL contains invalid column 'recommended_message'")
                else:
                    print(f"⚠️ UNKNOWN: Column pattern not found in SQL")
                    print(f"   Searching for 'recommend' in SQL: {'recommend' in sql.lower()}")
            else:
                print("⚠️ No SQL found in response")
            
            # Check execution results (try different possible keys)
            data = None
            if 'result' in result and result['result']:
                data = result['result']
            elif 'rows' in result and result['rows']:
                data = result['rows']
            elif 'data' in result and result['data']:
                data = result['data']
            
            if data:
                print(f"📈 Query returned {len(data)} rows")
                if data:
                    print(f"🔍 Sample columns: {list(data[0].keys())}")
                    # Show first few rows
                    for i, row in enumerate(data[:3]):
                        print(f"   Row {i+1}: {row}")
            else:
                print("📈 No data rows returned")
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_nba_query_with_fixed_schema()
