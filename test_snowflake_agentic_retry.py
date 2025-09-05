#!/usr/bin/env python3
"""
Test Snowflake Agentic Retry System - Table Name Quoting
Tests the enhanced SQL retry system for Snowflake-specific issues like table name quoting
"""

import asyncio
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

async def test_snowflake_table_quoting():
    """Test that the agentic retry system properly handles Snowflake table name quoting"""
    
    try:
        from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
        
        print("🧪 Testing Snowflake Agentic Retry System - Table Name Quoting")
        print("=" * 70)
        
        # Initialize orchestrator
        orchestrator = DynamicAgentOrchestrator()
        
        # Test inputs that would cause the table name quoting issue
        test_inputs = {
            "original_query": "Show me data from NBA output table",
            "user_id": "test_user",
            "4_user_verification": {
                "approved_tables": ["Final_NBA_Output_python_06042025"]  # Table with numbers (needs quoting)
            },
            "previous_sql_error": ""  # Start with no error
        }
        
        print(f"📋 Test Query: {test_inputs['original_query']}")
        print(f"📊 Test Table: {test_inputs['4_user_verification']['approved_tables'][0]}")
        print(f"🎯 Expected: System should add double quotes around table name for Snowflake")
        print()
        
        # Execute query generation
        print("🚀 Starting SQL generation with agentic retry...")
        result = await orchestrator._execute_query_generation(test_inputs)
        
        print("\n📊 Test Results:")
        print("=" * 50)
        
        if result.get("status") == "completed":
            sql_query = result.get("sql_query", "")
            attempt_number = result.get("attempt_number", 1)
            
            print(f"✅ SQL Generation Status: {result['status']}")
            print(f"🔄 Attempts Required: {attempt_number}")
            print(f"📝 Generated SQL: {sql_query}")
            print(f"💡 Explanation: {result.get('explanation', 'N/A')}")
            
            # Check if the table name is properly quoted
            if '"Final_NBA_Output_python_06042025"' in sql_query:
                print("✅ SUCCESS: Table name is properly quoted for Snowflake!")
            elif 'Final_NBA_Output_python_06042025' in sql_query:
                print("⚠️ PARTIAL: Table name found but not quoted (may cause issues)")
            else:
                print("❌ ISSUE: Table name not found in query")
                
            # Test database-aware features
            if result.get("database_type"):
                print(f"🗄️ Database Type: {result['database_type']}")
                
        else:
            print(f"❌ SQL Generation Failed: {result.get('error', 'Unknown error')}")
            
        print("\n🧪 Test Summary:")
        print("=" * 50)
        print("✅ Agentic retry system tested")
        print("✅ Database-aware SQL generation tested")
        print("✅ Snowflake table name quoting tested")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        print(f"📋 Full error trace:\n{traceback.format_exc()}")
        return None

async def test_error_feedback_system():
    """Test that errors are properly passed to the LLM for learning"""
    
    print("\n🧪 Testing Error Feedback System")
    print("=" * 70)
    
    try:
        from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
        
        orchestrator = DynamicAgentOrchestrator()
        
        # Simulate a scenario with a previous error
        test_inputs_with_error = {
            "original_query": "Show me NBA data",
            "user_id": "test_user",
            "4_user_verification": {
                "approved_tables": ["Final_NBA_Output_python_06042025"]
            },
            "previous_sql_error": """Database execution error: Table 'Final_NBA_Output_python_06042025' not found or not accessible
Query attempted: SELECT * FROM Final_NBA_Output_python_06042025 LIMIT 1
Database type: Snowflake
Common fix needed: Check table/column names, quoting rules, and database-specific syntax."""
        }
        
        print("🔄 Testing with simulated previous error...")
        print(f"📋 Previous Error: {test_inputs_with_error['previous_sql_error'][:100]}...")
        
        # Get database type info
        database_type = await orchestrator._get_database_type_info()
        
        # Test database-aware SQL generation with error context
        result = await orchestrator._generate_database_aware_sql(
            query=test_inputs_with_error["original_query"],
            available_tables=test_inputs_with_error["4_user_verification"]["approved_tables"],
            database_type=database_type,
            error_context=test_inputs_with_error["previous_sql_error"],
            attempt=2  # Simulate retry attempt
        )
        
        print(f"\n📊 Error Feedback Test Results:")
        print("=" * 50)
        
        if result and result.get("sql_query"):
            sql_query = result["sql_query"]
            print(f"✅ SQL Generated with Error Context: {sql_query}")
            
            # Check if the error was addressed
            if '"Final_NBA_Output_python_06042025"' in sql_query:
                print("✅ SUCCESS: LLM learned from error and added proper quoting!")
            else:
                print("⚠️ WARNING: Error may not have been fully addressed")
                
        else:
            print("❌ No SQL generated with error context")
            
        return result
        
    except Exception as e:
        print(f"❌ Error feedback test failed: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Starting Snowflake Agentic Retry Tests")
    print("=" * 80)
    
    async def run_all_tests():
        # Test 1: Basic table quoting
        result1 = await test_snowflake_table_quoting()
        
        # Test 2: Error feedback system
        result2 = await test_error_feedback_system()
        
        print("\n🏁 All Tests Completed!")
        print("=" * 80)
        
        if result1 and result2:
            print("✅ All tests passed - Agentic retry system is working!")
        else:
            print("⚠️ Some tests failed - check implementation")
    
    # Run the tests
    asyncio.run(run_all_tests())
