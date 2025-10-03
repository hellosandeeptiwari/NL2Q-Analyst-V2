#!/usr/bin/env python3
"""
Trigger Selective Reindexing using Backend API
This uses the existing backend functionality to properly reindex Azure SQL tables
"""
import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

async def trigger_selective_reindexing():
    """Use the backend's selective reindexing functionality"""
    
    print("🚀 Triggering Selective Reindexing for Azure SQL Tables")
    print(f"📋 Database Engine: {os.getenv('DB_ENGINE')}")
    print(f"🏢 Database: {os.getenv('AZURE_SQL_DATABASE')}")
    
    # Import the backend orchestrator
    from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
    
    # Azure tables that need proper indexing
    azure_tables = [
        'Reporting_BI_NGD',
        'Reporting_BI_PrescriberOverview', 
        'Reporting_BI_PrescriberProfile'
    ]
    
    print(f"🎯 Target tables: {azure_tables}")
    
    try:
        # Initialize orchestrator
        print("📊 Initializing orchestrator...")
        orchestrator = DynamicAgentOrchestrator()
        
        # Trigger selective reindexing with force_clear=True to ensure fresh data
        print("🔄 Starting selective reindexing with force_clear=True...")
        print("   This will:")
        print("   ✅ Clear existing vectors for these tables")
        print("   ✅ Fetch fresh schema from Azure SQL")
        print("   ✅ Create new vectors with full column information")
        
        await orchestrator._perform_full_database_indexing(
            force_clear=True,  # Clear existing data for fresh indexing
            selected_tables=azure_tables
        )
        
        print("🎉 Selective reindexing completed!")
        
        # Verify the results
        print("\n🔍 Verifying indexing results...")
        from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
        
        pinecone_store = PineconeSchemaVectorStore()
        
        for table in azure_tables:
            try:
                # Check if table is indexed with column information
                query_result = pinecone_store.index.query(
                    vector=[0.1] * 3072,
                    filter={'table_name': table},
                    top_k=1,
                    include_metadata=True
                )
                
                if query_result.matches:
                    metadata = query_result.matches[0].metadata
                    column_count = metadata.get('column_count', 'unknown')
                    columns = metadata.get('columns', [])
                    print(f"✅ {table}: Column count: {column_count}, Stored columns: {len(columns)}")
                    if len(columns) > 0:
                        print(f"   Sample columns: {columns[:5]}")
                    else:
                        print(f"   ⚠️ Still no column information - indexing may need additional fixes")
                else:
                    print(f"❌ {table}: No vectors found after indexing")
                    
            except Exception as e:
                print(f"❌ {table}: Error checking results - {e}")
        
        print("\n✅ Reindexing process completed!")
        print("🧪 Run the test again to see if column information is now available")
        
    except Exception as e:
        print(f"❌ Error during reindexing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(trigger_selective_reindexing())