#!/usr/bin/env python3
"""
Quick fix test - directly use NBA table when found
"""
import sys
import asyncio
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "backend"))

async def test_direct_nba_query():
    """Test using the NBA table directly"""
    
    print("🧪 Testing Direct NBA Query Processing...")
    
    query = "read table final nba output python and fetch top 5 rows and create a visualization with frequency of recommended message and provider input"
    
    try:
        # Step 1: Check if NBA table exists in Snowflake
        from backend.db.enhanced_schema import get_enhanced_schema_cache
        schema_cache = get_enhanced_schema_cache()
        
        print(f"📝 Query: {query}")
        
        # Find the NBA table directly
        nba_table = None
        for table in schema_cache.get('tables', []):
            if isinstance(table, dict) and 'NBA' in table.get('name', '').upper():
                nba_table = table
                break
        
        if nba_table:
            print(f"✅ Found NBA table: {nba_table['name']}")
            
            # Check for relevant columns
            columns = nba_table.get('columns', [])
            relevant_columns = []
            for col in columns:
                col_name = col.get('name', '') if isinstance(col, dict) else str(col)
                if any(keyword in col_name.lower() for keyword in ['message', 'provider', 'input', 'recommendation', 'output']):
                    relevant_columns.append(col_name)
            
            print(f"🎯 Relevant columns found: {len(relevant_columns)}")
            for col in relevant_columns[:10]:
                print(f"   • {col}")
            
            # Generate SQL query for this table
            sql_query = f"""
            SELECT 
                {', '.join(relevant_columns[:10]) if relevant_columns else '*'}
            FROM {nba_table['full_qualified_name'] if 'full_qualified_name' in nba_table else nba_table['name']}
            LIMIT 5
            """
            
            print(f"\n💾 Generated SQL:")
            print(sql_query)
            
            # Test the query execution
            try:
                from backend.db.engine import get_adapter
                adapter = get_adapter()
                
                print(f"\n⚡ Testing query execution...")
                result = adapter.run(sql_query, dry_run=False)
                
                if result.error:
                    print(f"❌ Query failed: {result.error}")
                else:
                    print(f"✅ Query successful: {len(result.rows)} rows returned")
                    if result.rows:
                        print(f"📊 Sample data: {result.rows[0][:5]}")
                
                return True
                
            except Exception as e:
                print(f"❌ Query execution failed: {e}")
                return False
                
        else:
            print(f"❌ NBA table not found in schema cache")
            print(f"📋 Available tables:")
            for table in schema_cache.get('tables', [])[:5]:
                if isinstance(table, dict):
                    print(f"   • {table.get('name', 'unknown')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_direct_nba_query())
