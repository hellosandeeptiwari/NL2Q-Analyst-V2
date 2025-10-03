#!/usr/bin/env python3
"""
Check Available Tables and Indexing Status
"""
import sys
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

async def check_table_status():
    from backend.db.engine import get_adapter
    from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
    
    try:
        print("=== Checking Available Tables in Azure SQL ===")
        adapter = get_adapter()  # Uses DB_ENGINE from .env
        
        print(f"Database engine: {os.getenv('DB_ENGINE')}")
        print(f"Database host: {os.getenv('AZURE_SQL_HOST')}")
        print(f"Database name: {os.getenv('AZURE_SQL_DATABASE')}")
        
        # Get all available tables
        print("\n=== Available Tables ===")
        
        try:
            sql = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'TABLE'"
            result = adapter.run(sql)
            if result.error:
                print(f"Error getting tables: {result.error}")
            else:
                tables = [row[0] for row in result.rows]
                print(f"Found {len(tables)} tables:")
                for table in sorted(tables):
                    print(f"  • {table}")
        except Exception as e:
            print(f"Error: {e}")
            
        # Check specific tables we know exist
        print("\n=== Checking Specific Tables ===")
        test_tables = ["Reporting_BI_NGD", "Reporting_BI_PrescriberOverview", "Reporting_BI_PrescriberProfile"]
        for table in test_tables:
            try:
                result = adapter.run(f"SELECT TOP 1 * FROM [{table}]")
                if result.error:
                    print(f"❌ {table}: {result.error}")
                else:
                    print(f"✅ {table}: {len(result.columns)} columns available")
                    print(f"   Sample columns: {result.columns[:5]}")
            except Exception as e:
                print(f"❌ {table}: {e}")
        
        # Check what's indexed in Pinecone
        print("\n=== Checking Pinecone Indexing Status ===")
        try:
            pinecone_store = PineconeSchemaVectorStore()
            indexed_tables = await pinecone_store._get_indexed_tables_fast()
            print(f"Indexed tables ({len(indexed_tables)}): {sorted(indexed_tables)}")
            
            # Check if our target tables are properly indexed
            for table in test_tables:
                if table in indexed_tables:
                    # Get a sample vector to check column information
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
                        print(f"✅ {table} indexed - Column count: {column_count}, Stored columns: {len(columns)}")
                        if len(columns) > 0:
                            print(f"   Sample columns: {columns[:5]}")
                        else:
                            print(f"   ⚠️ No column information stored in metadata")
                    else:
                        print(f"⚠️ {table} indexed but no vectors found")
                else:
                    print(f"❌ {table} NOT indexed")
                    
        except Exception as e:
            print(f"Error checking Pinecone: {e}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_table_status())