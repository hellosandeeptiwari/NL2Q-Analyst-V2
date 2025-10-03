#!/usr/bin/env python3
"""
Critical Root Cause Analysis - Data Retrieval Investigation
"""

import asyncio
import sys
sys.path.append('.')

from backend.tools.sql_runner import SQLRunner

async def diagnose_data_issue():
    print("🔍 CRITICAL ANALYSIS: Investigating data retrieval failure")
    print("=" * 60)
    
    sql_runner = SQLRunner()
    user_id = "diagnostic_user"
    
    # Test queries from simple to complex
    test_queries = [
        "SELECT 1 as test_value",  # Most basic test
        "SELECT COUNT(*) as table_count FROM INFORMATION_SCHEMA.TABLES",  # Schema info
        "SELECT COUNT(*) as total_rows FROM Reporting_BI_PrescriberOverview",  # Row count
        "SELECT TOP 1 PrescriberName FROM Reporting_BI_PrescriberOverview",  # Single row
        "SELECT TOP 3 PrescriberName, RegionName FROM Reporting_BI_PrescriberOverview"  # Multiple rows
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🧪 Test {i}: {query}")
        print("-" * 40)
        
        try:
            result = await sql_runner.execute_query(query, user_id=user_id)
            
            if result and hasattr(result, 'success') and result.success:
                data = result.data if hasattr(result, 'data') and result.data is not None else []
                print(f"✅ Success: {len(data)} rows returned")
                
                if data:
                    print(f"📊 First row: {data[0]}")
                    if len(data) > 1:
                        print(f"📊 Total rows: {len(data)}")
                else:
                    print("❌ Query executed but returned 0 rows")
                    
            else:
                error_msg = getattr(result, 'error_message', 'Unknown error')
                print(f"❌ Query failed: {error_msg}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    print("\n" + "=" * 60)
    print("🔍 DIAGNOSIS COMPLETE")

if __name__ == "__main__":
    asyncio.run(diagnose_data_issue())