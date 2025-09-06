from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
import asyncio

async def test_fixed_schema_extraction():
    print("Testing fixed schema extraction...\n")
    
    vector_store = PineconeSchemaVectorStore()
    table_name = "Final_NBA_Output_python_20250519"
    
    # Test the fixed get_table_details function
    table_details = await vector_store.get_table_details(table_name)
    
    print(f"🔍 Table Details for {table_name}:")
    print(f"   📋 Total Columns Found: {len(table_details.get('columns', []))}")
    print(f"   📋 Columns: {table_details.get('columns', [])}")
    print(f"   📋 Available Chunks: {list(table_details.get('chunks', {}).keys())}")
    
    # Check if we found the target column
    columns = table_details.get('columns', [])
    target_column = "Recommended_Msg_Overall"
    
    if target_column in columns:
        print(f"✅ SUCCESS: Found target column '{target_column}'!")
    else:
        print(f"❌ MISSING: Target column '{target_column}' not found")
        print(f"   Available columns: {columns}")

if __name__ == "__main__":
    asyncio.run(test_fixed_schema_extraction())
