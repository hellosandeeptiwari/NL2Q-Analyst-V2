#!/usr/bin/env python3
"""
Reindex Azure SQL Tables into Pinecone Vector Store
This will properly capture all column information for the 3 Azure tables
"""
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

async def reindex_azure_tables():
    print("🚀 Starting Azure SQL Tables Reindexing...")
    
    from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
    from backend.db.engine import get_adapter
    
    # Initialize with Azure SQL
    pinecone_store = PineconeSchemaVectorStore()
    db_adapter = get_adapter()  # Will use DB_ENGINE from .env (should be azure)
    
    print("✅ Pinecone and Azure SQL initialized")
    print(f"🔍 Database engine: {os.getenv('DB_ENGINE', 'not set')}")
    
    # Azure tables we need to reindex
    azure_tables = [
        'Reporting_BI_NGD',
        'Reporting_BI_PrescriberOverview', 
        'Reporting_BI_PrescriberProfile'
    ]
    
    print(f"🎯 Target Azure tables: {azure_tables}")
    
    # Clear existing vectors for these tables first
    print("🧹 Clearing existing vectors for Azure tables...")
    for table in azure_tables:
        try:
            pinecone_store.index.delete(filter={'table_name': table})
            print(f"  ✅ Cleared vectors for {table}")
        except Exception as e:
            print(f"  ⚠️ Could not clear {table}: {e}")
    
    # Reindex with full column information
    print("📊 Starting fresh indexing with full column details...")
    await pinecone_store.index_database_schema(
        db_adapter, 
        selected_tables=azure_tables
    )
    
    print("🎉 Azure tables reindexing complete!")
    
    # Verify the indexing worked
    print("\n🔍 Verifying indexed content...")
    for table in azure_tables:
        try:
            query_result = pinecone_store.index.query(
                vector=[0.1] * 3072,
                filter={'table_name': table},
                top_k=1,
                include_metadata=True
            )
            
            if query_result.matches:
                match = query_result.matches[0]
                metadata = match.metadata
                print(f"✅ {table}:")
                print(f"  - Chunks found: {len(query_result.matches)}")
                print(f"  - Column count: {metadata.get('column_count', 'unknown')}")
                print(f"  - Columns stored: {len(metadata.get('columns', []))}")
                print(f"  - Sample columns: {metadata.get('columns', [])[:5]}")
            else:
                print(f"❌ {table}: No vectors found")
                
        except Exception as e:
            print(f"❌ {table}: Error verifying - {e}")
    
    print("\n✅ Reindexing verification complete!")

if __name__ == "__main__":
    asyncio.run(reindex_azure_tables())