from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
import asyncio

async def test_target_table_details():
    print("Testing target table details...\n")
    
    vector_store = PineconeSchemaVectorStore()
    table_name = "Final_NBA_Output_python_06042025"
    
    try:
        details = await vector_store.get_table_details(table_name)
        
        print(f"✅ Table: {details['table_name']}")
        print(f"Chunks available: {list(details.get('chunks', {}).keys())}")
        
        # Check each column group
        for chunk_type, chunk_data in details.get('chunks', {}).items():
            if chunk_type == 'column_group':
                metadata = chunk_data.get('metadata', {})
                columns = metadata.get('columns', [])
                group_name = metadata.get('column_group', 'unknown')
                print(f"\n📋 Column Group: {group_name}")
                print(f"   Columns: {columns}")
                
                # Check for our target column
                if 'Recommended_Msg_Overall' in columns:
                    print(f"   🎯 FOUND 'Recommended_Msg_Overall' in this group!")
                    
        # Now test a query that should find this table
        print(f"\n🔍 Testing query for 'show recommended messages'...")
        matches = await vector_store.search_relevant_tables("show recommended messages", top_k=5)
        
        for i, match in enumerate(matches):
            if match['table_name'] == table_name:
                print(f"  ✅ Found our table at position {i+1} with score {match['total_score']:.3f}")
                break
        else:
            print(f"  ❌ Our table not found in top 5 results")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_target_table_details())
