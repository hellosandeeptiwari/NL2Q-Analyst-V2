"""
Test script to verify table and column comments are being retrieved
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test_comments_retrieval():
    """Test that comments are retrieved from Azure SQL"""
    from backend.db.engine import get_adapter
    
    print("=" * 70)
    print("🔍 Testing Table and Column Comments Retrieval")
    print("=" * 70)
    
    # Get adapter
    adapter = get_adapter()
    
    # Get a sample table
    tables = await adapter.get_table_names()
    print(f"\n📊 Total tables: {len(tables)}")
    
    # Show first 5 tables with comments
    print(f"\n📋 First 5 tables with their comments:")
    for i, table in enumerate(tables[:5]):
        table_name = table['name']
        table_comment = table.get('comment', None)
        print(f"\n{i+1}. Table: {table_name}")
        print(f"   Schema: {table['schema']}")
        if table_comment:
            print(f"   📝 Comment: {table_comment}")
        else:
            print(f"   ⚠️  No comment found")
        
        # Get columns for this table
        columns = await adapter.get_table_schema(table_name, table['schema'])
        print(f"   📊 Total columns: {len(columns)}")
        
        # Show first 3 columns with comments
        for j, col in enumerate(columns[:3]):
            col_name = col['name']
            col_type = col['type']
            col_comment = col.get('comment', None)
            print(f"      • {col_name} ({col_type})", end="")
            if col_comment:
                print(f" - 📝 {col_comment}")
            else:
                print(" - ⚠️ No comment")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    asyncio.run(test_comments_retrieval())
