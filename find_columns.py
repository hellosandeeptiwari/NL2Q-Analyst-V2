from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
import asyncio
import json

async def find_tables_with_columns():
    print("Finding tables with actual column information...\n")
    
    vector_store = PineconeSchemaVectorStore()
    
    # Search for various tables to see if any have column info
    query = "columns schema table"
    matches = await vector_store.search_relevant_tables(query, top_k=10)
    
    for i, match in enumerate(matches[:5]):
        table_name = match['table_name']
        print(f"\n{i+1}. Checking {table_name}...")
        
        try:
            details = await vector_store.get_table_details(table_name)
            
            has_columns = False
            for chunk_type, chunk_data in details['chunks'].items():
                metadata = chunk_data.get('metadata', {})
                
                if 'columns' in metadata or 'column_names' in metadata:
                    has_columns = True
                    print(f"  ✅ {chunk_type}: Has column info!")
                    if 'columns' in metadata:
                        cols = metadata['columns']
                        if isinstance(cols, list):
                            print(f"     Columns: {cols[:5]}{'...' if len(cols) > 5 else ''}")
                        else:
                            print(f"     Columns: {cols}")
                elif chunk_type == 'column_group' or 'column' in chunk_type.lower():
                    print(f"  📋 {chunk_type}: {chunk_data.get('content', 'N/A')[:100]}...")
                    has_columns = True
                    
            if not has_columns:
                print(f"  ❌ No column information found")
                
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    asyncio.run(find_tables_with_columns())
