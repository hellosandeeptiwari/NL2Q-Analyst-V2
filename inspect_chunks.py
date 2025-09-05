from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
import asyncio
import json

async def inspect_table_chunks():
    print("Inspecting chunks for Final_NBA_Output_python_06042025...\n")
    
    vector_store = PineconeSchemaVectorStore()
    
    try:
        details = await vector_store.get_table_details('Final_NBA_Output_python_06042025')
        
        print(f"Table details: {details['table_name']}")
        print(f"Metadata: {details['metadata']}")
        print("\nChunks:")
        
        for chunk_type, chunk_data in details['chunks'].items():
            print(f"\n  {chunk_type}:")
            print(f"    Content: {chunk_data.get('content', 'N/A')[:300]}...")
            print(f"    Metadata: {chunk_data.get('metadata', {})}")
            
            # Look specifically for column information
            metadata = chunk_data.get('metadata', {})
            if 'columns' in metadata:
                print(f"    ✅ COLUMNS FOUND: {metadata['columns']}")
            elif 'column_names' in metadata:
                print(f"    ✅ COLUMN NAMES FOUND: {metadata['column_names']}")
            elif 'schema' in metadata:
                print(f"    ✅ SCHEMA FOUND: {metadata['schema']}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(inspect_table_chunks())
