#!/usr/bin/env python3
"""
Direct Pinecone clear and reindex with correct Azure SQL schema
"""

import sys
import os
from pathlib import Path

# Add the backend directory to Python path
script_dir = Path(__file__).parent
backend_dir = script_dir / "backend"
sys.path.insert(0, str(backend_dir))

def main():
    print("🔄 Direct Pinecone clear and reindex with correct Azure SQL schema")
    print("=" * 80)
    
    try:
        # Import after path setup
        from vector_store.pinecone_client import PineconeClient
        from database.database_adapter_factory import get_adapter
        from indexing.schema_indexer import SchemaIndexer
        
        # Initialize components
        print("🔌 Initializing components...")
        db_adapter = get_adapter()
        print(f"✅ Database adapter: {db_adapter.__class__.__name__}")
        
        pinecone_client = PineconeClient()
        print(f"✅ Pinecone client: {pinecone_client.index_name}")
        
        schema_indexer = SchemaIndexer(db_adapter, pinecone_client)
        print(f"✅ Schema indexer initialized")
        
        # 1. Clear existing vectors
        print("\n🗑️ Step 1: Clearing all vectors from Pinecone...")
        try:
            # Delete all vectors
            stats = pinecone_client.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            print(f"   Current vectors: {total_vectors}")
            
            if total_vectors > 0:
                # Delete all vectors by namespace (if using namespaces) or delete all
                pinecone_client.index.delete(delete_all=True)
                print(f"   ✅ Deleted all vectors")
            else:
                print(f"   ✅ Index already empty")
                
        except Exception as e:
            print(f"   ❌ Error clearing vectors: {e}")
            return
        
        # 2. Get actual Azure SQL table schemas
        print("\n📊 Step 2: Getting actual Azure SQL schemas...")
        tables = ['Reporting_BI_NGD', 'Reporting_BI_PrescriberOverview', 'Reporting_BI_PrescriberProfile']
        
        for table in tables:
            print(f"\n   📋 Processing {table}...")
            try:
                # Get actual columns from database
                columns = db_adapter.get_table_schema(table)
                print(f"      ✅ Found {len(columns)} actual columns")
                print(f"      📝 First 10: {columns[:10]}")
                
                # Index this table with correct schema
                result = schema_indexer.index_table_schema(table, force_reindex=True)
                print(f"      🎯 Indexing result: {result}")
                
            except Exception as e:
                print(f"      ❌ Error processing {table}: {e}")
        
        # 3. Verify the reindexing
        print("\n🔍 Step 3: Verifying reindexing...")
        try:
            stats = pinecone_client.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            print(f"   📊 Total vectors after reindex: {total_vectors}")
            
            # Test search to see if columns are correct
            search_results = pinecone_client.search("territory performance rep names", top_k=3)
            print(f"   🔍 Test search returned {len(search_results.get('matches', []))} matches")
            
            for i, match in enumerate(search_results.get('matches', [])[:2]):
                metadata = match.get('metadata', {})
                table_name = metadata.get('table_name', 'unknown')
                columns = metadata.get('columns', [])
                print(f"      {i+1}. {table_name}: {len(columns)} columns - {columns[:5]}...")
                
        except Exception as e:
            print(f"   ❌ Error verifying: {e}")
        
        print("\n✅ Direct reindex completed!")
        print("🎯 Now test the retry logic with correct schema")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()