#!/usr/bin/env python3
"""
Debug script to compare actual Azure SQL table columns vs. Pinecone indexed columns
"""

import sys
import os
import logging
from pathlib import Path

# Add the backend directory to Python path
script_dir = Path(__file__).parent
backend_dir = script_dir / "backend"
sys.path.insert(0, str(backend_dir))

from database.database_adapter_factory import get_adapter
from vector_store.pinecone_client import PineconeClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("🔍 Comparing actual Azure SQL columns vs. Pinecone indexed columns")
    print("=" * 80)
    
    try:
        # Get the database adapter
        db_adapter = get_adapter()
        print(f"✅ Using database adapter: {db_adapter.__class__.__name__}")
        
        # Initialize Pinecone
        pinecone_client = PineconeClient()
        print(f"✅ Connected to Pinecone index: {pinecone_client.index_name}")
        
        # Test tables
        tables = ['Reporting_BI_NGD', 'Reporting_BI_PrescriberOverview', 'Reporting_BI_PrescriberProfile']
        
        for table in tables:
            print(f"\n📊 TABLE: {table}")
            print("-" * 50)
            
            # Get actual columns from Azure SQL
            try:
                actual_columns = db_adapter.get_table_schema(table)
                print(f"🔗 Actual Azure SQL columns ({len(actual_columns)}):")
                for i, col in enumerate(actual_columns[:10]):  # Show first 10
                    print(f"  {i+1:2d}. {col}")
                if len(actual_columns) > 10:
                    print(f"  ... and {len(actual_columns) - 10} more columns")
                
            except Exception as e:
                print(f"❌ Could not get actual columns: {e}")
                actual_columns = []
            
            # Get Pinecone indexed columns
            try:
                # Search for this table in Pinecone
                search_results = pinecone_client.search(f"table {table}", top_k=5)
                
                pinecone_columns = set()
                for match in search_results.get('matches', []):
                    metadata = match.get('metadata', {})
                    table_name = metadata.get('table_name', '')
                    if table_name == table:
                        columns = metadata.get('columns', [])
                        pinecone_columns.update(columns)
                
                pinecone_columns = sorted(list(pinecone_columns))
                print(f"🎯 Pinecone indexed columns ({len(pinecone_columns)}):")
                for i, col in enumerate(pinecone_columns[:10]):  # Show first 10
                    print(f"  {i+1:2d}. {col}")
                if len(pinecone_columns) > 10:
                    print(f"  ... and {len(pinecone_columns) - 10} more columns")
                
            except Exception as e:
                print(f"❌ Could not get Pinecone columns: {e}")
                pinecone_columns = []
            
            # Compare columns
            if actual_columns and pinecone_columns:
                actual_set = set(actual_columns)
                pinecone_set = set(pinecone_columns)
                
                missing_in_pinecone = actual_set - pinecone_set
                extra_in_pinecone = pinecone_set - actual_set
                
                print(f"\n🔍 COMPARISON:")
                print(f"  ✅ Matching columns: {len(actual_set & pinecone_set)}")
                print(f"  ❌ Missing in Pinecone: {len(missing_in_pinecone)}")
                print(f"  ⚠️  Extra in Pinecone: {len(extra_in_pinecone)}")
                
                if missing_in_pinecone:
                    print(f"  Missing: {sorted(list(missing_in_pinecone))[:5]}...")
                if extra_in_pinecone:
                    print(f"  Extra: {sorted(list(extra_in_pinecone))[:5]}...")
        
        print(f"\n🔍 Testing specific problematic columns:")
        print("-" * 50)
        
        # Test the specific columns that failed
        problem_columns = {
            'Reporting_BI_NGD': ['Territory', 'PerformanceMetric', 'RepID', 'TerritoryName'],
            'Reporting_BI_PrescriberOverview': ['RepName', 'RepID', 'PrescriberID'],
            'Reporting_BI_PrescriberProfile': ['PrescriberID']
        }
        
        for table, cols in problem_columns.items():
            print(f"\n📊 {table} - Problem columns:")
            try:
                actual_columns = db_adapter.get_table_schema(table)
                for col in cols:
                    exists = col in actual_columns
                    status = "✅" if exists else "❌"
                    print(f"  {status} {col}")
            except Exception as e:
                print(f"  ❌ Could not check columns: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()