import asyncio
import sys
import os
sys.path.append('backend')

from backend.tools.sql_runner import SQLRunner

async def test_sql_runner():
    print("=== Testing SQL Runner with Real Database ===")
    
    try:
        runner = SQLRunner()
        print("✅ SQL Runner created successfully")
        
        # Test the exact query that was failing
        sql = 'SELECT * FROM "Final_NBA_Output_python_06042025" LIMIT 3'
        print(f"🔍 Testing query: {sql}")
        
        result = await runner.execute_query(
            sql=sql,
            user_id="test_user",
            timeout_seconds=30,
            max_rows=10
        )
        
        print(f"📊 Query result:")
        print(f"  Success: {result.success}")
        print(f"  Error: {result.error_message}")
        print(f"  Row count: {result.row_count}")
        print(f"  Columns: {result.columns}")
        print(f"  Execution time: {result.execution_time}")
        
        if result.success and result.data:
            print(f"  Sample data: {result.data[0]}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_sql_runner())
    print(f"\n🎯 Test {'PASSED' if success else 'FAILED'}")
