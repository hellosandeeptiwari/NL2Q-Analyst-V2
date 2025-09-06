import requests
import json

def test_fixed_traditional():
    print("🔧 Testing FIXED traditional endpoint...\n")
    
    query = "What are the recommended messages for NBA marketing actions?"
    payload = {
        "natural_language": query,
        "job_id": "test_fixed",
        "db_type": "snowflake"
    }
    
    try:
        response = requests.post("http://localhost:8000/query", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Fixed Traditional endpoint SUCCESS")
            sql = result.get('sql', 'No SQL found')
            rows = result.get('rows', [])
            
            print(f"🔧 Generated SQL: {sql}")
            print(f"📈 Returned {len(rows)} rows")
            
            # Check if SQL contains actual target tables (not NBA_outputs)
            target_tables = ['Final_NBA_Output_python_20250502', 'Final_NBA_Output_python_070125', 'Final_NBA_Output_python_06262025_w_HCP_Call']
            
            found_correct_table = False
            for table_name in target_tables:
                if table_name in sql:
                    print(f"✅ SQL contains correct target table: {table_name}")
                    found_correct_table = True
                    break
            
            if not found_correct_table:
                print(f"❌ SQL STILL using wrong table")
                print(f"   Expected one of: {target_tables}")
                print(f"   Got: {sql}")
                
                # Check if it's still using NBA_outputs
                if 'NBA_outputs' in sql:
                    print(f"❌ CRITICAL: Still using NBA_outputs instead of Pinecone results!")
                else:
                    print(f"❌ Using some other unexpected table")
            else:
                print(f"🎉 SUCCESS: Traditional endpoint now using correct tables!")
                
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
    test_fixed_traditional()
