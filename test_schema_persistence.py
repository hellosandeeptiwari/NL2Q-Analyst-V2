import requests
import json

def test_schema_persistence():
    print("🔍 Testing schema cache persistence and traditional endpoint...\n")
    
    # Step 1: Refresh schema cache
    print("=" * 60)
    print("STEP 1: Refresh schema cache")
    print("=" * 60)
    
    try:
        refresh_response = requests.get("http://localhost:8000/refresh-schema")
        if refresh_response.status_code == 200:
            refresh_result = refresh_response.json()
            tables_count = refresh_result.get('tables_count', 0)
            target_tables = refresh_result.get('target_tables', [])
            print(f"✅ Schema refreshed: {tables_count} tables")
            print(f"🎯 Target tables: {target_tables[:3]}...")
        else:
            print(f"❌ Schema refresh failed: {refresh_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Schema refresh error: {e}")
        return
    
    # Step 2: Immediately check schema cache
    print(f"\n" + "=" * 60)
    print("STEP 2: Verify schema cache immediately")
    print("=" * 60)
    
    try:
        schema_response = requests.get("http://localhost:8000/schema")
        if schema_response.status_code == 200:
            schema = schema_response.json()
            print(f"✅ Schema cache contains {len(schema)} tables")
            
            # Check for our target tables
            target_table_found = False
            for table_name in target_tables[:3]:
                if table_name in schema:
                    print(f"✅ Found {table_name} in schema cache")
                    columns = schema[table_name]
                    if 'Recommended_Msg_Overall' in columns:
                        print(f"   🎯 Has target column!")
                        target_table_found = True
                else:
                    print(f"❌ Missing {table_name} from schema cache")
            
            if not target_table_found:
                print(f"❌ No target tables found in schema cache")
                return
                
        else:
            print(f"❌ Schema check failed: {schema_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Schema check error: {e}")
        return
    
    # Step 3: Immediately test traditional endpoint
    print(f"\n" + "=" * 60)
    print("STEP 3: Test traditional endpoint immediately")
    print("=" * 60)
    
    query = "What are the recommended messages for NBA marketing actions?"
    payload = {
        "natural_language": query,
        "job_id": "test_persistence",
        "db_type": "snowflake"
    }
    
    try:
        response = requests.post("http://localhost:8000/query", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Traditional endpoint SUCCESS")
            sql = result.get('sql', 'No SQL found')
            rows = result.get('rows', [])
            
            print(f"🔧 Generated SQL: {sql}")
            print(f"📈 Returned {len(rows)} rows")
            
            # Check if SQL contains target tables
            found_target = False
            for table_name in target_tables:
                if table_name in sql:
                    print(f"✅ SQL contains target table: {table_name}")
                    found_target = True
                    break
            
            if not found_target:
                print(f"❌ SQL doesn't contain target tables")
                print(f"   Expected one of: {target_tables}")
                print(f"   Got: {sql}")
                
            # Check for target column
            if 'Recommended_Msg_Overall' in sql:
                print(f"✅ SQL contains target column!")
            else:
                print(f"❌ SQL missing target column")
                
        else:
            print(f"❌ Traditional endpoint failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Traditional endpoint error: {e}")

if __name__ == "__main__":
    test_schema_persistence()
