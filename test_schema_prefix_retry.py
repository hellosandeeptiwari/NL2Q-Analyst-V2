#!/usr/bin/env python3
"""
Test script to validate the enhanced retry logic handles schema prefix issues
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator

load_dotenv()

async def test_retry_with_schema_error():
    """Test if the retry logic can fix schema prefix issues"""
    
    print("🧪 Testing Enhanced Retry Logic - Schema Prefix Fix")
    print("=" * 60)
    
    # Initialize the orchestrator
    orchestrator = DynamicAgentOrchestrator()
    
    # Test query that should trigger the retry logic
    query = "Show me territories with the lowest performance, including rep names"
    
    print(f"📝 Query: {query}")
    print(f"🎯 Expected: Should fix schema prefix issues and execute successfully")
    print(f"🔄 Testing enhanced error correction with schema metadata...")
    print()
    
    try:
        # This should trigger the full pipeline including retry logic
        result = await orchestrator.process_query(
            user_query=query,
            user_id="test_user",
            session_id="test_session"
        )
        
        print("🔍 RESULT ANALYSIS:")
        print("=" * 40)
        
        if result.get('status') == 'success':
            print("✅ SUCCESS: Query executed successfully!")
            
            # Check if SQL was corrected
            if 'sql_query' in result:
                sql = result['sql_query']
                print(f"📊 Final SQL (first 200 chars): {sql[:200]}...")
                
                # Check for corrections
                if 'dbo.' not in sql:
                    print("✅ Schema prefix was correctly removed!")
                else:
                    print("⚠️ Schema prefix still present - may need further refinement")
                    
            # Check results
            if 'data' in result and result['data']:
                print(f"📈 Retrieved {len(result['data'])} rows of data")
                print("✅ End-to-end pipeline working!")
            else:
                print("⚠️ No data returned - possible execution issue")
                
        else:
            print("❌ FAILED: Query execution failed")
            if 'error' in result:
                print(f"💥 Error: {result['error']}")
                
            # This is where retry logic should have kicked in
            print("🔍 Checking if retry was attempted...")
            if 'retry_attempts' in result:
                print(f"🔄 Retry attempts made: {result['retry_attempts']}")
            else:
                print("⚠️ No retry information found")
        
    except Exception as e:
        print(f"💥 Exception during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_retry_with_schema_error())