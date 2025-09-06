import requests
import json
import sys

def debug_query_pipeline():
    print("🔍 Debugging complete query pipeline...\n")
    
    # Test query
    query = "What are the recommended messages for NBA marketing actions?"
    
    # Step 1: Test direct schema lookup
    print("=" * 60)
    print("STEP 1: Testing direct schema lookup")
    print("=" * 60)
    
    try:
        from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
        import asyncio
        
        async def test_schema_search():
            vector_store = PineconeSchemaVectorStore()
            
            # Search for relevant tables
            search_results = await vector_store.search_relevant_tables(query, top_k=5)
            print(f"📋 Found {len(search_results)} relevant tables:")
            
            for i, result in enumerate(search_results):
                table_name = result.get('table_name', 'unknown')
                score = result.get('score', 0)
                print(f"   {i+1}. {table_name} (score: {score:.3f})")
                
                # Get detailed schema for top result
                if i == 0:
                    table_details = await vector_store.get_table_details(table_name)
                    columns = table_details.get('columns', [])
                    print(f"      📊 Columns ({len(columns)}): {columns}")
            
            return search_results
        
        search_results = asyncio.run(test_schema_search())
        
    except Exception as e:
        print(f"❌ Schema lookup failed: {e}")
        search_results = []
    
    # Step 2: Test full API call with detailed logging
    print("\n" + "=" * 60)
    print("STEP 2: Testing full API call")
    print("=" * 60)
    
    url = "http://localhost:8000/query"
    payload = {
        "question": query,
        "user_id": "test_user",
        "session_id": "debug_session"
    }
    
    try:
        print(f"📡 Sending query to {url}")
        response = requests.post(url, json=payload, timeout=60)
        
        print(f"🌐 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"📊 Response keys: {list(result.keys())}")
            
            # Examine SQL
            sql = result.get('sql', 'No SQL found')
            print(f"\n🔧 Generated SQL:")
            print("-" * 40)
            print(sql)
            print("-" * 40)
            
            # Check for schema information in the response
            if 'suggestions' in result:
                suggestions = result['suggestions']
                print(f"\n💡 Suggestions: {suggestions}")
            
            # Check bias report for schema info
            if 'bias_report' in result:
                bias_report = result['bias_report']
                print(f"\n📝 Bias Report: {bias_report}")
            
            # Analyze the SQL
            if "NULL AS" in sql and "No Data Available" in sql:
                print("\n❌ PROBLEM: LLM generated fallback SQL - no schema reached the LLM")
                print("   This means the orchestrator didn't find relevant tables or")
                print("   the schema extraction pipeline failed.")
            elif "Recommended_Msg_Overall" in sql:
                print("\n✅ SUCCESS: SQL contains correct column name!")
            elif "recommended_message" in sql.lower():
                print("\n⚠️ PARTIAL: SQL contains incorrect column name")
            else:
                print("\n❓ UNCLEAR: SQL doesn't contain expected column patterns")
                
        else:
            print(f"❌ API call failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ API call failed: {e}")

if __name__ == "__main__":
    debug_query_pipeline()
