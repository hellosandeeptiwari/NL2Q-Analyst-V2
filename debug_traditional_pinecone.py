import requests
import json

def debug_pinecone_search_in_traditional():
    print("🔍 Debugging Pinecone search in traditional endpoint...\n")
    
    # Let's test Pinecone search directly first
    print("=" * 60)
    print("STEP 1: Testing Pinecone search directly")
    print("=" * 60)
    
    try:
        from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
        import asyncio
        
        async def test_pinecone_search():
            vector_store = PineconeSchemaVectorStore()
            query = "What are the recommended messages for NBA marketing actions?"
            
            # Search for relevant tables
            search_results = await vector_store.search_relevant_tables(query, top_k=3)
            print(f"📋 Found {len(search_results)} relevant tables:")
            
            for i, result in enumerate(search_results):
                table_name = result.get('table_name', 'unknown')
                score = result.get('total_score', 0)
                print(f"   {i+1}. {table_name} (score: {score:.3f})")
                
                # Get detailed schema for each result
                try:
                    table_details = await vector_store.get_table_details(table_name)
                    columns = table_details.get('columns', [])
                    print(f"      📊 Columns ({len(columns)}): {columns[:5]}{'...' if len(columns) > 5 else ''}")
                    
                    # Check for target column
                    if 'Recommended_Msg_Overall' in columns:
                        print(f"      🎯 HAS TARGET COLUMN!")
                except Exception as e:
                    print(f"      ❌ Error getting details: {e}")
            
            return search_results
        
        search_results = asyncio.run(test_pinecone_search())
        
        if search_results:
            print(f"\n✅ Pinecone search working - found {len(search_results)} relevant tables")
        else:
            print(f"\n❌ Pinecone search returned no results")
            return
        
    except Exception as e:
        print(f"❌ Pinecone search failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("STEP 2: Testing traditional endpoint with debug")
    print("=" * 60)
    
    # Now test the traditional endpoint
    query = "What are the recommended messages for NBA marketing actions?"
    
    payload = {
        "natural_language": query,
        "job_id": "debug_traditional",
        "db_type": "snowflake"
    }
    
    try:
        print(f"📡 Sending query to traditional endpoint...")
        response = requests.post("http://localhost:8000/query", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Traditional endpoint responded")
            sql = result.get('sql', 'No SQL found')
            print(f"🔧 Generated SQL: {sql}")
            
            # Check if the SQL contains any of our target tables
            target_tables = [res.get('table_name', '') for res in search_results]
            found_target = False
            for table in target_tables:
                if table in sql:
                    print(f"✅ SQL contains target table: {table}")
                    found_target = True
                    break
            
            if not found_target:
                print(f"❌ SQL doesn't contain any target tables: {target_tables}")
                print(f"   Instead, it's looking for: {sql}")
                print(f"   This suggests the schema context didn't reach the LLM properly")
                
        else:
            print(f"❌ Traditional endpoint failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Traditional endpoint error: {e}")

if __name__ == "__main__":
    debug_pinecone_search_in_traditional()
