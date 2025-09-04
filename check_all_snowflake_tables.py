#!/usr/bin/env python3
"""
Check All Available Tables in Snowflake
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

def check_all_tables():
    """Check all tables available in Snowflake"""
    
    print("📊 Checking All Available Tables in Snowflake...")
    
    try:
        from backend.db.engine import get_adapter
        
        # Get Snowflake adapter
        adapter = get_adapter("snowflake")
        
        # Get all tables
        result = adapter.run("SHOW TABLES IN SCHEMA ENHANCED_NBA", dry_run=False)
        
        if result.error:
            print(f"❌ Failed to get tables: {result.error}")
            return
        
        print(f"✅ Found {len(result.rows)} tables in ENHANCED_NBA schema:")
        
        # Filter for NBA-related tables
        nba_tables = []
        other_tables = []
        
        for row in result.rows:
            table_name = row[1] if len(row) > 1 else str(row[0])  # Get table name
            
            if "NBA" in table_name.upper():
                nba_tables.append(table_name)
            else:
                other_tables.append(table_name)
        
        print(f"\n🏀 NBA-related tables ({len(nba_tables)}):")
        for i, table in enumerate(nba_tables):
            print(f"   {i+1}. {table}")
        
        print(f"\n📋 Other tables ({len(other_tables)}):")
        for i, table in enumerate(other_tables[:10]):  # Show first 10
            print(f"   {i+1}. {table}")
        
        if len(other_tables) > 10:
            print(f"   ... and {len(other_tables) - 10} more")
        
        # Test specific NBA query matching
        print(f"\n🔍 Tables matching 'final nba output python':")
        query_terms = ["final", "nba", "output", "python"]
        
        matching_tables = []
        for table in result.rows:
            table_name = row[1] if len(row) > 1 else str(row[0])
            table_lower = table_name.lower()
            
            # Count matching terms
            match_count = sum(1 for term in query_terms if term in table_lower)
            
            if match_count >= 2:  # At least 2 terms match
                matching_tables.append((table_name, match_count))
        
        # Sort by match count
        matching_tables.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   Found {len(matching_tables)} tables with 2+ matching terms:")
        for i, (table, count) in enumerate(matching_tables[:10]):
            print(f"   {i+1}. {table} (matches: {count})")
        
        return {
            "all_tables": [row[1] if len(row) > 1 else str(row[0]) for row in result.rows],
            "nba_tables": nba_tables,
            "matching_tables": [table for table, count in matching_tables]
        }
        
    except Exception as e:
        print(f"❌ Failed to check tables: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    tables = check_all_tables()
    
    if tables:
        print(f"\n📊 SUMMARY:")
        print(f"   Total tables: {len(tables['all_tables'])}")
        print(f"   NBA tables: {len(tables['nba_tables'])}")
        print(f"   Query matches: {len(tables['matching_tables'])}")
    else:
        print(f"\n❌ Failed to get table information")
