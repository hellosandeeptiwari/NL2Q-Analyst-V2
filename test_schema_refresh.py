#!/usr/bin/env python3
"""
Test script to refresh enhanced schema cache and verify datatypes are properly captured
"""

import sys
import os
import json
sys.path.append('.')

from backend.db.enhanced_schema import get_enhanced_schema_cache

def test_schema_refresh():
    """Test refreshing the enhanced schema cache and check datatypes"""
    print("🔄 Refreshing enhanced schema cache...")
    
    # This will force a fresh build since we deleted the cache
    schema = get_enhanced_schema_cache()
    
    print(f"✅ Schema refreshed with {len(schema.get('tables', []))} tables")
    
    # Check a few sample tables for proper datatypes
    tables = schema.get('tables', [])
    
    for table in tables[:3]:  # Check first 3 tables
        table_name = table.get('name', 'Unknown')
        columns = table.get('columns', [])
        
        print(f"\n📊 Table: {table_name}")
        print(f"   Columns: {len(columns)}")
        
        # Check datatypes for first few columns
        for col in columns[:5]:  # First 5 columns
            col_name = col.get('name', 'Unknown')
            data_type = col.get('data_type', 'MISSING')
            print(f"   - {col_name}: {data_type}")
            
            if data_type == 'unknown':
                print(f"      ⚠️ Column {col_name} has unknown datatype!")
    
    # Check if TirosintTargetFlag exists and has proper datatype
    pharma_tables = [t for t in tables if 'pharma' in t.get('name', '').lower() or 'tirosint' in t.get('name', '').lower()]
    
    if pharma_tables:
        print(f"\n🧪 Found {len(pharma_tables)} pharma-related tables")
        for table in pharma_tables:
            columns = table.get('columns', [])
            tirosint_cols = [c for c in columns if 'tirosint' in c.get('name', '').lower()]
            if tirosint_cols:
                print(f"   Table {table.get('name')}: Found {len(tirosint_cols)} Tirosint columns")
                for col in tirosint_cols:
                    print(f"     - {col.get('name')}: {col.get('data_type')}")

if __name__ == "__main__":
    test_schema_refresh()