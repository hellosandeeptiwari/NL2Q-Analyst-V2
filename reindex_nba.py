from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
from backend.db.engine import get_adapter
import asyncio

async def test_reindex_nba_table():
    print("Testing re-indexing NBA table with fresh column data...\n")
    
    # Get actual columns from database
    adapter = get_adapter()
    table_name = "Final_NBA_Output_python_06042025"
    
    try:
        result = adapter.run(f'DESCRIBE TABLE "COMMERCIAL_AI"."ENHANCED_NBA"."{table_name}"', dry_run=False)
        if result.error:
            print(f"❌ Error getting columns: {result.error}")
            return
        
        print(f"📊 Found {len(result.rows)} columns in database:")
        columns = {}
        for row in result.rows:
            col_name = row[0]
            col_type = row[1]
            columns[col_name] = {"data_type": col_type, "nullable": False}
            print(f"  - {col_name} ({col_type})")
        
        # Test the column grouping logic
        vector_store = PineconeSchemaVectorStore()
        column_groups = vector_store._group_columns_by_purpose(columns)
        
        print(f"\n🔗 Column groups identified:")
        for group_name, group_cols in column_groups.items():
            print(f"  {group_name}: {list(group_cols.keys())}")
        
        # Create table info dict with correct structure
        table_info = {
            'name': table_name,  # Changed from table_name to name!
            'schema': "ENHANCED_NBA",
            'columns': columns,
            'row_count': None
        }
        
        # Test chunk creation
        chunks = vector_store.chunk_schema_information(table_info)
        
        print(f"\n📦 Created {len(chunks)} chunks:")
        for chunk in chunks:
            print(f"  - {chunk.chunk_type}: {chunk.chunk_id}")
            if chunk.chunk_type == 'column_group':
                print(f"    Columns in metadata: {chunk.metadata.get('columns', [])}")
                
        # Now let's delete and re-index just this table
        print(f"\n🔄 Re-indexing {table_name}...")
        
        # Delete existing chunks for this table
        vector_store.index.delete(
            filter={"table_name": table_name}
        )
        print(f"✅ Deleted existing chunks for {table_name}")
        
        # Re-index with fresh data
        vectors = []
        for chunk in chunks:
            if chunk.content and chunk.content.strip():
                try:
                    # Generate embedding
                    embedding = await vector_store.generate_embedding(chunk.content)
                    
                    # Create vector for upsert (clean metadata)
                    clean_metadata = {
                        "table_name": chunk.table_name,
                        "table_schema": chunk.table_schema,
                        "chunk_type": chunk.chunk_type,
                        "content": chunk.content
                    }
                    
                    # Add non-null metadata values
                    for key, value in chunk.metadata.items():
                        if value is not None:
                            clean_metadata[key] = value
                    
                    vector = {
                        "id": chunk.chunk_id,
                        "values": embedding,
                        "metadata": clean_metadata
                    }
                    vectors.append(vector)
                    print(f"    ✅ Created vector for {chunk.chunk_type}")
                except Exception as e:
                    print(f"    ❌ Failed to create vector for {chunk.chunk_type}: {e}")
        
        # Upsert vectors
        if vectors:
            vector_store.index.upsert(vectors=vectors)
            print(f"✅ Uploaded {len(vectors)} vectors for {table_name}")
        else:
            print(f"❌ No vectors created for {table_name}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_reindex_nba_table())
