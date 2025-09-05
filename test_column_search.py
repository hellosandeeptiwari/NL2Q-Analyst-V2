from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
import asyncio

async def test_column_search():
    print("Testing column-specific search...\n")
    
    vector_store = PineconeSchemaVectorStore()
    
    queries = [
        "Recommended_Msg_Overall",
        "recommended message overall column",
        "NBA recommendation column"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        matches = await vector_store.search_relevant_tables(query, top_k=5)
        
        for i, match in enumerate(matches):
            print(f"  {i+1}. {match['table_name']} (score: {match['total_score']:.3f})")
            
            # Check if this is our target table
            if 'unknown' in match['table_name'] or 'Final_NBA_Output_python_06042025' in match['table_name']:
                print(f"    🎯 Found target! Sample: {match['sample_content']}")
                
                try:
                    details = await vector_store.get_table_details(match['table_name'])
                    print(f"    Chunks: {list(details.get('chunks', {}).keys())}")
                    
                    for chunk_type, chunk_data in details.get('chunks', {}).items():
                        if chunk_type == 'column_group':
                            metadata = chunk_data.get('metadata', {})
                            if 'columns' in metadata:
                                columns = metadata['columns']
                                if 'Recommended_Msg_Overall' in columns:
                                    print(f"    ✅ FOUND 'Recommended_Msg_Overall' in {chunk_type}!")
                                print(f"    Columns: {columns}")
                except Exception as e:
                    print(f"    Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_column_search())
