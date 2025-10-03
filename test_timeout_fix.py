#!/usr/bin/env python3
"""
Test the timeout fix for hanging SQL execution
"""

import asyncio
import sys
sys.path.append('.')

async def test_timeout_fix():
    print("🔧 TESTING TIMEOUT FIX FOR HANGING SQL")
    print("=" * 50)
    
    from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
    
    try:
        orchestrator = DynamicAgentOrchestrator()
        
        # Test the same query that was hanging
        query = "Show me territories with the lowest performance, including rep names"
        print(f"📝 Query: {query}")
        print("⏰ This should now timeout after 30 seconds instead of hanging indefinitely")
        
        result = await orchestrator.process_query(query)
        
        print(f"\n📊 Result status: {result.get('status')}")
        if result.get('status') == 'failed':
            error = result.get('error', '')
            if 'timed out' in error.lower():
                print("✅ SUCCESS: Timeout handling working - no more infinite hanging!")
                print(f"🔍 Error message: {error}")
            else:
                print(f"⚠️ Different error: {error}")
        elif result.get('sql'):
            print("✅ SUCCESS: Query completed successfully!")
            print(f"🔍 Generated SQL: {result['sql'][:200]}...")
        else:
            print(f"⚠️ Unexpected result: {result}")
            
    except Exception as e:
        print(f"❌ Error testing timeout fix: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_timeout_fix())