import requests
import json

def test_prompt_fix():
    print("🔧 Testing if LLM prompt fix works...\n")
    
    # Test the traditional endpoint
    query = "What are the recommended messages for NBA marketing actions?"
    payload = {
        "natural_language": query,
        "job_id": "test_prompt_fix",
        "db_type": "snowflake"
    }
    
    try:
        response = requests.post("http://localhost:8000/query", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            sql = result.get('sql', 'No SQL found')
            rows = result.get('rows', [])
            
            print(f"✅ Traditional endpoint responded")
            print(f"🔧 Generated SQL: {sql}")
            print(f"📈 Returned {len(rows)} rows")
            
            # Check if SQL contains the correct tables (not NBA_outputs)
            if 'NBA_outputs' in sql:
                print(f"❌ STILL using wrong table: NBA_outputs")
                print(f"   This means the LLM is still ignoring the schema")
            elif any(table in sql for table in ['Final_NBA_Output_python_20250502', 'Final_NBA_Output_python_070125', 'Final_NBA_Output_python_06262025_w_HCP_Call']):
                print(f"✅ SUCCESS! Using correct table from Pinecone search")
            else:
                print(f"❓ Using different table: {sql}")
                
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
    test_prompt_fix()
