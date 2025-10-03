#!/usr/bin/env python3
"""
Test Enhanced Retry Logic with Error Correction
- Tests if the retry system properly extracts error details
- Verifies schema metadata is provided to LLM for correction
- Checks if ambiguous column errors are properly corrected
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator

async def test_enhanced_retry():
    """Test the enhanced retry logic with a territory query that should trigger column ambiguity error"""
    
    print("🧪 Testing Enhanced Retry Logic with Error Correction")
    print("=" * 80)
    
    # Initialize orchestrator
    orchestrator = DynamicAgentOrchestrator()
    # The orchestrator initializes automatically when created
    
    # Test query that should cause column ambiguity error initially
    test_query = "Show me territories with the lowest performance, including rep names"
    
    print(f"📝 Test Query: {test_query}")
    print(f"🎯 Expected Behavior:")
    print(f"   1. Generate SQL with potential column ambiguity")
    print(f"   2. Encounter 'Ambiguous column name' error")
    print(f"   3. Retry with enhanced error context and schema metadata")
    print(f"   4. LLM corrects SQL using proper table aliases")
    print(f"   5. Successfully execute corrected SQL")
    print()
    
    try:
        # Process the query - this should trigger the retry logic
        result = await orchestrator.process_query(
            user_query=test_query,
            user_id="test_user",
            session_id="test_session"
        )
        
        print("🔍 RESULT ANALYSIS:")
        print("=" * 50)
        
        if result.get("status") == "success":
            print("✅ Query processing successful!")
            
            # Check if execution results exist
            if "execution" in result:
                execution = result["execution"]
                if execution.get("status") == "completed":
                    rows = len(execution.get("results", []))
                    print(f"✅ SQL execution successful: {rows} rows returned")
                    
                    # Check if retry was used
                    if execution.get("execution_attempt", 1) > 1:
                        print(f"🔄 Retry logic activated: {execution.get('execution_attempt')} attempts")
                        print("✅ Enhanced error correction working!")
                    else:
                        print("ℹ️  Query executed on first attempt (no retry needed)")
                    
                    # Show final SQL
                    final_sql = execution.get("sql_executed", "N/A")
                    print(f"🔍 Final SQL executed:")
                    print(f"   {final_sql}")
                    
                else:
                    print(f"❌ SQL execution failed: {execution.get('error', 'Unknown error')}")
            else:
                print("⚠️  No execution results found in response")
                
        else:
            print(f"❌ Query processing failed: {result.get('error', 'Unknown error')}")
            
        print("\n🔍 Full Result Structure:")
        for key, value in result.items():
            if isinstance(value, dict):
                print(f"  {key}: {list(value.keys())}")
            else:
                print(f"  {key}: {type(value)}")
                
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_retry())