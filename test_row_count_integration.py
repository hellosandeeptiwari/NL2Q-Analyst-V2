from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
from backend.db.engine import get_adapter
import asyncio
import json

async def test_row_count_integration():
    print("Testing row count integration in schema reading...\n")
    
    vector_store = PineconeSchemaVectorStore()
    adapter = get_adapter()
    
    # Test with a known table
    table_name = "Final_NBA_Output_python_06042025"
    
    try:
        print(f"🔍 Getting table info for {table_name}...")
        table_info = await vector_store._get_table_info(adapter, table_name)
        
        print(f"✅ Table info retrieved:")
        print(f"   Name: {table_info.get('name')}")
        print(f"   Columns: {len(table_info.get('columns', {}))}")
        print(f"   Row Count: {table_info.get('row_count')}")
        
        # Test chunking with row counts
        print(f"\n📦 Testing schema chunking...")
        chunks = vector_store.chunk_schema_information(table_info)
        
        for chunk in chunks:
            if chunk.chunk_type == 'table_overview':
                print(f"\n🔍 Table Overview Chunk:")
                print(f"   Content preview: {chunk.content[:200]}...")
                print(f"   Row count in metadata: {chunk.metadata.get('row_count')}")
                print(f"   Table size category: {chunk.metadata.get('table_size_category')}")
                
                # Check if row count is in the content
                if 'Row Count:' in chunk.content:
                    print(f"   ✅ Row count included in embedding content!")
                else:
                    print(f"   ❌ Row count missing from embedding content")
                break
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_row_count_integration())
