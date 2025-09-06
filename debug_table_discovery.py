import requests
import json

def debug_table_discovery_process():
    print("🔍 Debugging the complete table discovery process...\n")
    
    # We'll examine what tables are being discovered and passed through the pipeline
    query = "What are the recommended messages for NBA marketing actions?"
    
    url = "http://localhost:8000/query"
    payload = {
        "question": query,
        "user_id": "test_user", 
        "session_id": "debug_table_discovery"
    }
    
    print(f"📡 Sending query: {query}")
    print(f"🔍 Looking for specific debug output in backend logs...")
    print("   - Schema discovery results")
    print("   - User verification/approval step")
    print("   - SQL generation with schema context")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ Query completed successfully")
            print(f"📊 Response structure: {list(result.keys())}")
            
            # Check if this contains the orchestrator's nested results
            if 'results' in result:
                orchestrator_results = result['results']
                print(f"\n🔍 Orchestrator results: {list(orchestrator_results.keys())}")
                
                # Look for schema discovery results
                if '1_discover_schema' in orchestrator_results:
                    schema_result = orchestrator_results['1_discover_schema']
                    print(f"\n📋 Schema Discovery Results:")
                    print(f"   Tables found: {schema_result.get('pinecone_matches', [])}")
                
                # Look for user verification results
                if '4_user_verification' in orchestrator_results:
                    verification_result = orchestrator_results['4_user_verification']
                    print(f"\n👤 User Verification Results:")
                    print(f"   Approved tables: {verification_result.get('approved_tables', [])}")
                
                # Look for SQL generation results
                if '5_generate_query' in orchestrator_results:
                    sql_result = orchestrator_results['5_generate_query']
                    print(f"\n🔧 SQL Generation Results:")
                    print(f"   SQL: {sql_result.get('sql_query', 'No SQL found')}")
                    print(f"   Status: {sql_result.get('status', 'Unknown')}")
                    
                    sql = sql_result.get('sql_query', '')
                    if "Recommended_Msg_Overall" in sql:
                        print(f"   ✅ SUCCESS: Contains correct column!")
                    elif "recommended_message" in sql.lower():
                        print(f"   ❌ FAILURE: Contains wrong column!")
                    elif "SELECT NULL" in sql:
                        print(f"   ⚠️ FALLBACK: Using NULL fallback SQL")
            
            # Also check the main response SQL
            main_sql = result.get('sql', result.get('sql_query', ''))
            if main_sql:
                print(f"\n🔧 Main Response SQL: {main_sql}")
        
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    debug_table_discovery_process()
