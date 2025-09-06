import requests
import json

def test_with_debug_output():
    print("🔧 Testing traditional endpoint to see debug output...")
    
    # Make request to capture debug output
    query = "What are the recommended messages for NBA marketing actions?"
    payload = {
        "natural_language": query,
        "job_id": "debug_schema_format",
        "db_type": "snowflake"
    }
    
    try:
        response = requests.post("http://localhost:8000/query", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            sql = result.get('sql', 'No SQL found')
            
            print(f"\n✅ Traditional endpoint responded")
            print(f"🔧 Generated SQL: {sql}")
            
            # Check what table the LLM used
            if 'NBA_outputs' in sql:
                print(f"❌ LLM used wrong table: NBA_outputs")
            elif any(table in sql for table in ['Final_NBA_Output_python_20250502', 'Final_NBA_Output_python_070125']):
                print(f"✅ LLM used correct table from schema!")
            else:
                print(f"❓ LLM used unknown table")
                
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_with_debug_output()
