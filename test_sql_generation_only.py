#!/usr/bin/env python3
"""
Test Snowflake SQL Generation - Table Name Quoting
Tests just the SQL generation without database execution
"""

import asyncio
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

async def test_sql_generation_only():
    """Test just the SQL generation without database execution"""
    
    try:
        from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
        
        print("🧪 Testing Snowflake SQL Generation - Table Name Quoting")
        print("=" * 70)
        
        # Initialize orchestrator
        orchestrator = DynamicAgentOrchestrator()
        
        # Get database type info
        database_type = await orchestrator._get_database_type_info()
        print(f"🗄️ Database Engine: {database_type['engine']}")
        print(f"🗄️ Database Dialect: {database_type['dialect'].name}")
        print(f"🗄️ Quote Character: {database_type['syntax_guide']['quote_char']}")
        print()
        
        # Test case 1: Table with numbers (should be quoted)
        print("📋 Test Case 1: Table with numbers")
        result1 = await orchestrator._generate_database_aware_sql(
            query="Show me data from NBA output table",
            available_tables=["Final_NBA_Output_python_06042025"],
            database_type=database_type,
            error_context="",
            attempt=1
        )
        
        if result1 and result1.get("sql_query"):
            sql1 = result1["sql_query"]
            print(f"✅ Generated SQL: {sql1}")
            if '"Final_NBA_Output_python_06042025"' in sql1:
                print("✅ SUCCESS: Table name is properly quoted!")
            else:
                print("❌ FAIL: Table name is not quoted")
        else:
            print("❌ No SQL generated")
        
        print()
        
        # Test case 2: With error context (retry scenario)
        print("📋 Test Case 2: With error feedback (retry scenario)")
        error_context = """Database execution error: Table 'Final_NBA_Output_python_06042025' not found or not accessible
Query attempted: SELECT * FROM Final_NBA_Output_python_06042025 LIMIT 1
Database type: Snowflake

Common issues for Snowflake:
- Table/column names with numbers need proper quoting
- Case sensitivity rules
- Database-specific function names
- Permission/access issues

Please fix the query considering these database-specific requirements."""
        
        result2 = await orchestrator._generate_database_aware_sql(
            query="Show me data from NBA output table",
            available_tables=["Final_NBA_Output_python_06042025"],
            database_type=database_type,
            error_context=error_context,
            attempt=2
        )
        
        if result2 and result2.get("sql_query"):
            sql2 = result2["sql_query"]
            print(f"✅ Generated SQL with error context: {sql2}")
            if '"Final_NBA_Output_python_06042025"' in sql2:
                print("✅ SUCCESS: LLM learned from error and added proper quoting!")
            else:
                print("❌ FAIL: LLM did not learn from error feedback")
        else:
            print("❌ No SQL generated with error context")
        
        print()
        print("🧪 Test Summary:")
        print("=" * 50)
        
        success_count = 0
        if result1 and '"Final_NBA_Output_python_06042025"' in result1.get("sql_query", ""):
            success_count += 1
        if result2 and '"Final_NBA_Output_python_06042025"' in result2.get("sql_query", ""):
            success_count += 1
            
        print(f"✅ Tests passed: {success_count}/2")
        
        if success_count == 2:
            print("🎉 All tests passed! Agentic SQL generation is working correctly.")
        elif success_count == 1:
            print("⚠️ Partial success - check error feedback system.")
        else:
            print("❌ Tests failed - SQL generation needs improvement.")
        
        return success_count == 2
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        print(f"📋 Full error trace:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Snowflake SQL Generation Test")
    print("=" * 80)
    
    # Run the test
    success = asyncio.run(test_sql_generation_only())
    
    if success:
        print("\n🏁 Test completed successfully!")
    else:
        print("\n💥 Test failed - check implementation")
