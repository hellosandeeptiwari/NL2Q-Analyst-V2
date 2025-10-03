#!/usr/bin/env python3
"""
Script to refresh Pinecone schema index with updated datatypes
"""

import sys
import os
import asyncio
sys.path.append('.')

from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
from backend.db.engine import get_adapter

async def refresh_pinecone_schema():
    """Refresh the Pinecone schema index with updated datatypes"""
    print("🧹 Refreshing Pinecone schema index...")
    
    # Initialize components
    vector_store = PineconeSchemaVectorStore()
    db_adapter = get_adapter()
    
    # Get a few key tables to reindex (focus on pharma tables)
    key_tables = [
        "PharmaGCO_DetailedAnalysis", 
        "PharmaGCO_AllSamples",
        "patient_data",
        "treatment_outcomes"
    ]
    
    print(f"🎯 Reindexing {len(key_tables)} key tables with fresh schema...")
    
    # Delete existing entries for these tables first
    for table_name in key_tables:
        try:
            # Get existing vectors for this table
            query_response = vector_store.index.query(
                filter={"table_name": table_name},
                top_k=1000,  # Get all vectors for this table
                include_metadata=True
            )
            
            if query_response.matches:
                vector_ids = [match.id for match in query_response.matches]
                print(f"   🗑️ Deleting {len(vector_ids)} old vectors for {table_name}")
                vector_store.index.delete(ids=vector_ids)
            else:
                print(f"   ℹ️ No existing vectors found for {table_name}")
                
        except Exception as e:
            print(f"   ⚠️ Error cleaning up {table_name}: {e}")
    
    # Reindex with fresh schema
    try:
        await vector_store.index_database_schema(
            db_adapter=db_adapter,
            selected_tables=key_tables
        )
        print("✅ Pinecone schema refresh completed!")
        
        # Test retrieving a table to verify datatypes
        print("\n🔍 Testing datatype retrieval...")
        table_info = await vector_store._get_table_info(db_adapter, "PharmaGCO_DetailedAnalysis")
        
        if table_info and 'columns' in table_info:
            columns = table_info['columns']
            print(f"📊 PharmaGCO_DetailedAnalysis has {len(columns)} columns:")
            
            # Check first few columns for datatypes
            for col_name, col_info in list(columns.items())[:5]:
                data_type = col_info.get('data_type', 'MISSING')
                print(f"   - {col_name}: {data_type}")
                
                if data_type == 'unknown':
                    print(f"     ⚠️ Still showing unknown datatype!")
                    
            # Specifically check for TirosintTargetFlag
            if 'TirosintTargetFlag' in columns:
                tirosint_type = columns['TirosintTargetFlag'].get('data_type', 'MISSING')
                print(f"   🎯 TirosintTargetFlag: {tirosint_type}")
            else:
                print("   ❌ TirosintTargetFlag not found in columns")
        
    except Exception as e:
        print(f"❌ Error during reindexing: {e}")

if __name__ == "__main__":
    asyncio.run(refresh_pinecone_schema())