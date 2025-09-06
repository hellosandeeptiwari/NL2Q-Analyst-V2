from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
import asyncio

async def debug_post_reingest():
    print("Debugging Pinecone after reingest...\n")
    
    vector_store = PineconeSchemaVectorStore()
    
    # Check if our target table now has proper schema
    table_name = "Final_NBA_Output_python_20250519"  # From the error log
    
    print(f"🔍 Searching for {table_name}...")
    matches = await vector_store.search_relevant_tables(table_name, top_k=5)
    
    for i, match in enumerate(matches):
        if table_name in match['table_name']:
            print(f"\n✅ Found {match['table_name']} at position {i+1}")
            print(f"   Score: {match['total_score']:.3f}")
            
            # Get detailed schema
            try:
                details = await vector_store.get_table_details(match['table_name'])
                print(f"   Available chunks: {list(details.get('chunks', {}).keys())}")
                
                # Check column groups
                for chunk_type, chunk_data in details.get('chunks', {}).items():
                    if chunk_type == 'column_group':
                        metadata = chunk_data.get('metadata', {})
                        columns = metadata.get('columns', [])
                        group_name = metadata.get('column_group', 'unknown')
                        print(f"   📋 {group_name}: {columns}")
                        
                        # Check for our target columns
                        if 'Recommended_Msg_Overall' in columns:
                            print(f"      🎯 Found 'Recommended_Msg_Overall'!")
                        if 'input_id' in columns:
                            print(f"      🎯 Found 'input_id'!")
                            
            except Exception as e:
                print(f"   ❌ Error getting details: {e}")
                
    # Also test the schema extraction function directly
    print(f"\n🔧 Testing schema extraction for query...")
    query = "show recommended messages"
    matches = await vector_store.search_relevant_tables(query, top_k=4)
    
    print(f"Found {len(matches)} matches for '{query}':")
    for i, match in enumerate(matches):
        print(f"  {i+1}. {match['table_name']} (score: {match['total_score']:.3f})")
        
        # Test the get_table_details function that the orchestrator uses
        if i == 0:  # Test first match
            try:
                details = await vector_store.get_table_details(match['table_name'])
                has_columns = False
                for chunk_type, chunk_data in details.get('chunks', {}).items():
                    if chunk_type == 'column_group':
                        columns = chunk_data.get('metadata', {}).get('columns', [])
                        if columns:
                            has_columns = True
                            print(f"    ✅ Has columns: {columns[:3]}{'...' if len(columns) > 3 else ''}")
                            break
                if not has_columns:
                    print(f"    ❌ No column information available")
            except Exception as e:
                print(f"    ❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug_post_reingest())
