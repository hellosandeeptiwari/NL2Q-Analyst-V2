#!/usr/bin/env python3
"""
Test to create a working territories query using actual available columns.
Now that we can access real column names, let's build a working performance query.
"""

# Simple working query based on real column names we discovered:
working_sql = """
SELECT TOP 10
    t1.[TerritoryName] AS [Territory],
    t2.[PrescriberName] AS [Representative],
    ISNULL(t2.[TRX(C4 Wk)], 0) AS [CurrentTransactions],
    ISNULL(t2.[NRX(C4 Wk)], 0) AS [NewPrescriptions],
    ISNULL(t2.[Calls4], 0) AS [CallActivity]
FROM
    [Reporting_BI_NGD] AS t1
JOIN
    [Reporting_BI_PrescriberOverview] AS t2 ON t1.[TerritoryId] = t2.[TerritoryId]
WHERE
    t2.[TRX(C4 Wk)] IS NOT NULL
ORDER BY
    t2.[TRX(C4 Wk)] ASC, t2.[NRX(C4 Wk)] ASC
"""

print("🎯 WORKING TERRITORIES QUERY using REAL columns:")
print("=" * 60)
print(working_sql)
print("=" * 60)
print()
print("✅ Key improvements from fallback mechanism:")
print("   • Uses real column names: TerritoryName, PrescriberName")
print("   • Uses real join column: TerritoryId")
print("   • Uses actual performance metrics: TRX(C4 Wk), NRX(C4 Wk)")
print("   • Properly handles nullable values with ISNULL")
print()
print("🚀 This should work because we now have the REAL database schema!")

# Test the query directly
import os
import sys
sys.path.append(os.path.dirname(__file__))

try:
    from backend.db.engine import get_adapter
    
    print("\n🧪 Testing the working query...")
    
    # Get database adapter
    db_adapter = get_adapter()
    print(f"✅ Got database adapter: {type(db_adapter).__name__}")
    
    # Test connection
    import asyncio
    
    async def test_query():
        try:
            # Execute the working query
            from backend.orchestrators.dynamic_agent_orchestrator import DatabaseConnector
            
            connector = DatabaseConnector()
            await connector.initialize_connection()
            
            result = await connector.execute_sql(working_sql)
            
            if result.success:
                print(f"🎉 SUCCESS! Query returned {result.row_count} rows")
                print("📊 Sample data:")
                for i, row in enumerate(result.data[:3]):
                    print(f"   Row {i+1}: {dict(zip(result.columns, row))}")
            else:
                print(f"❌ Query failed: {result.error_message}")
                
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
    
    # Run the async test
    asyncio.run(test_query())
    
except ImportError as e:
    print(f"⚠️ Could not import modules for testing: {e}")
    print("💡 But the SQL query above should work with the real columns!")