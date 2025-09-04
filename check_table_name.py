#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path('.').resolve() / 'backend'))

from backend.db.enhanced_schema import get_enhanced_schema_cache

schema = get_enhanced_schema_cache()
print("🔍 Checking NBA table name format...")

for table in schema.get('tables', []):
    if isinstance(table, dict) and 'NBA' in table.get('name', '').upper():
        print(f"Table name: {table.get('name')}")
        print(f"Full qualified: {table.get('full_qualified_name')}")
        print(f"Schema info: {table.get('schema')}")
        print(f"Database: {schema.get('database')}")
        print(f"Engine: {schema.get('engine')}")
        break
