"""
Debug script to see what table_info structure looks like during indexing
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def main():
    from backend.db.engine import AzureSQLAdapter
    from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore, get_database_config
    
    # Initialize database adapter
    print("✅ Initializing Azure SQL adapter...")
    config = get_database_config()
    db_adapter = AzureSQLAdapter(config)
    
    # Initialize Pinecone vector store
    print("✅ Initializing Pinecone vector store...")
    vector_store = PineconeSchemaVectorStore()
    
    # Get table info for our test table
    table_name = "Reporting_BI_TerritoryPerformanceSummary"
    print(f"\n🔍 Getting table info for {table_name}...")
    table_info = await vector_store._get_table_info(db_adapter, table_name)
    
    print(f"\n📊 TABLE INFO STRUCTURE:")
    print(f"  Table Name: {table_info.get('name')}")
    print(f"  Description: {table_info.get('description')}")
    print(f"  Total Columns: {len(table_info.get('columns', {}))}")
    
    print(f"\n📋 FIRST 5 COLUMNS DETAILED STRUCTURE:")
    columns = table_info.get('columns', {})
    for i, (col_name, col_info) in enumerate(list(columns.items())[:5]):
        print(f"\n  Column {i+1}: {col_name}")
        print(f"    Full info: {col_info}")
        print(f"    data_type: {col_info.get('data_type')}")
        print(f"    comment: {col_info.get('comment')}")
        print(f"    description: {col_info.get('description')}")
        print(f"    nullable: {col_info.get('nullable')}")
    
    # Check specifically for TRX column (we know it has a comment)
    if 'TRX' in columns:
        print(f"\n🎯 SPECIFIC CHECK FOR 'TRX' COLUMN:")
        trx_info = columns['TRX']
        print(f"  Full info: {trx_info}")
        print(f"  Has 'comment' field: {'comment' in trx_info}")
        print(f"  Comment value: {trx_info.get('comment')}")
        print(f"  Has 'description' field: {'description' in trx_info}")
        print(f"  Description value: {trx_info.get('description')}")
    
    # Now test the chunking to see what gets generated
    print(f"\n🔨 TESTING CHUNK GENERATION:")
    chunks = vector_store.chunk_schema_information(table_info)
    
    print(f"  Total chunks created: {len(chunks)}")
    
    # Find the metrics column group chunk
    for chunk in chunks:
        if chunk.chunk_type == "column_group" and "Metrics" in chunk.content:
            print(f"\n📦 METRICS COLUMN GROUP CHUNK:")
            print(f"  Chunk Type: {chunk.chunk_type}")
            print(f"  Content (first 600 chars):")
            print(f"  {chunk.content[:600]}")
            
            # Check if TRX line has comment
            for line in chunk.content.split('\n'):
                if 'TRX' in line and '•' in line:
                    print(f"\n  🔍 TRX LINE IN CHUNK: {line}")
            break
    
    db_adapter.close()

if __name__ == "__main__":
    asyncio.run(main())
