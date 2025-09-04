#!/usr/bin/env python3
"""
Check discovered tables in Snowflake schema cache
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.db.enhanced_schema import get_enhanced_schema_cache
import json

def check_discovered_tables():
    print("🔍 Checking discovered Snowflake tables...")
    
    try:
        schema = get_enhanced_schema_cache()
        
        print(f"📊 Schema cache structure:")
        print(f"   Keys: {list(schema.keys())}")
        
        if 'tables' in schema:
            tables = schema['tables']
            print(f"\n📋 Found {len(tables)} tables:")
            
            for i, table in enumerate(tables, 1):
                if isinstance(table, dict):
                    table_name = table.get('name', 'unknown')
                    print(f"   {i}. {table_name}")
                    
                    # Check if this is the NBA table
                    if 'NBA' in table_name.upper():
                        print(f"      ✅ NBA TABLE FOUND!")
                        print(f"      📝 Full name: {table_name}")
                        
                        # Show some columns
                        columns = table.get('columns', [])
                        if columns:
                            print(f"      🔧 Columns ({len(columns)}): {[col.get('name', 'unknown') for col in columns[:5]]}")
                else:
                    print(f"   {i}. {table}")
        else:
            print("❌ No 'tables' key found in schema")
            
        # Also check direct table keys
        table_keys = [k for k in schema.keys() if k not in ['engine', 'database', 'allowed_schemas', 'tables']]
        if table_keys:
            print(f"\n📋 Direct table entries: {len(table_keys)}")
            for table_key in table_keys[:10]:  # Show first 10
                print(f"   • {table_key}")
                if 'NBA' in table_key.upper():
                    print(f"     ✅ NBA TABLE FOUND!")
                    
    except Exception as e:
        print(f"❌ Error checking schema: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_discovered_tables()
