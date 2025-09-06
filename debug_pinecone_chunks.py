from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
import asyncio

async def debug_pinecone_chunks():
    print("Debugging actual Pinecone chunk content...\n")
    
    vector_store = PineconeSchemaVectorStore()
    table_name = "Final_NBA_Output_python_20250519"
    
    # Query Pinecone directly for this table's chunks
    dummy_vector = [0.0] * 3072
    results = vector_store.index.query(
        vector=dummy_vector,
        filter={"table_name": {"$eq": table_name}}, 
        top_k=20, 
        include_metadata=True
    )
    
    print(f"🔍 Found {len(results.matches)} chunks for {table_name}:")
    
    for i, match in enumerate(results.matches):
        metadata = match.metadata
        chunk_type = metadata.get("chunk_type", "unknown")
        
        print(f"\n{i+1}. Chunk Type: {chunk_type}")
        print(f"   ID: {match.id}")
        print(f"   Score: {match.score}")
        
        # Check for column information
        if chunk_type == "column_group":
            columns = metadata.get("columns", [])
            group_name = metadata.get("column_group", "unknown")
            print(f"   📋 Column Group: {group_name}")
            print(f"   📋 Columns: {columns}")
            
            if not columns:
                print(f"   ❌ NO COLUMNS FOUND - checking all metadata keys:")
                for key, value in metadata.items():
                    print(f"      {key}: {value}")
        
        elif chunk_type == "table_overview":
            row_count = metadata.get("row_count", "unknown")
            total_columns = metadata.get("total_columns", "unknown")
            print(f"   📊 Row Count: {row_count}")
            print(f"   📊 Total Columns: {total_columns}")
            
        # Check content preview
        content = metadata.get("content", "")
        if content:
            print(f"   Content preview: {content[:150]}...")

if __name__ == "__main__":
    asyncio.run(debug_pinecone_chunks())
