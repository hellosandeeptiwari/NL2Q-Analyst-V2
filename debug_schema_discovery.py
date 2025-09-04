#!/usr/bin/env python3
"""
Debug the schema discovery step specifically
"""
import sys
import asyncio
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "backend"))

async def debug_schema_discovery():
    """Debug what's happening in schema discovery"""
    
    print("🔍 Debugging Schema Discovery Step...")
    
    query = "read table final nba output python and fetch top 5 rows and create a visualization with frequency of recommended message and provider input"
    
    try:
        from backend.tools.schema_tool import SchemaTool
        schema_tool = SchemaTool()
        
        print(f"📝 Query: {query}")
        print(f"🔄 Running schema discovery...")
        
        # Run schema discovery
        schema_context = await schema_tool.discover_schema(query)
        
        print(f"\n📊 Schema Discovery Results:")
        print(f"   Relevant tables: {len(schema_context.relevant_tables)}")
        
        for i, table in enumerate(schema_context.relevant_tables, 1):
            print(f"   {i}. {table.name}")
            print(f"      Schema: {table.schema}")
            print(f"      Type: {table.type}")
            print(f"      Columns: {len(table.columns)}")
            print(f"      Description: {table.description}")
            
            if "NBA" in table.name.upper():
                print(f"      ✅ NBA TABLE FOUND!")
                # Show some key columns
                nba_columns = [col.name for col in table.columns if any(keyword in col.name.lower() for keyword in ['message', 'provider', 'input', 'recommendation'])]
                print(f"      🎯 Relevant columns: {nba_columns[:10]}")
        
        print(f"\n🏷️ Entity Mappings: {schema_context.entity_mappings}")
        print(f"📈 Metrics Available: {len(schema_context.metrics_available)}")
        print(f"📅 Date Columns: {len(schema_context.date_columns)}")
        
        return schema_context
        
    except Exception as e:
        print(f"❌ Schema discovery debug failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(debug_schema_discovery())
