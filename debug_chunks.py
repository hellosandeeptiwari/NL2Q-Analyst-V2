from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
import asyncio

async def debug_all_chunks():
    print("Debugging all chunks for our table...\n")
    
    vector_store = PineconeSchemaVectorStore()
    table_name = "Final_NBA_Output_python_06042025"
    
    # Query Pinecone directly for all chunks of this table
    index = vector_store.index
    
    # Fetch vectors by metadata filter
    try:
        query_response = index.query(
            vector=[0.0] * 3072,  # dummy vector
            filter={"table_name": table_name},
            top_k=10,
            include_metadata=True
        )
        
        print(f"Found {len(query_response.matches)} chunks for {table_name}:")
        
        for i, match in enumerate(query_response.matches):
            metadata = match.metadata
            chunk_type = metadata.get('chunk_type', 'unknown')
            print(f"\n{i+1}. Chunk: {chunk_type}")
            print(f"   ID: {match.id}")
            
            if chunk_type == 'column_group':
                columns = metadata.get('columns', [])
                group_name = metadata.get('column_group', 'unknown')
                print(f"   Group: {group_name}")
                print(f"   Columns: {columns}")
                
                if 'Recommended_Msg_Overall' in columns:
                    print(f"   🎯 FOUND 'Recommended_Msg_Overall'!")
            elif chunk_type == 'table_overview':
                print(f"   Content preview: {metadata.get('content', '')[:100]}...")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug_all_chunks())
