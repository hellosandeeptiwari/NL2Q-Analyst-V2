#!/usr/bin/env python3

import asyncio
import os
from backend.orchestrator.main_orchestrator import DynamicAgentOrchestrator

async def test_sql_extraction():
    """Quick test to debug SQL extraction issue"""
    print("Testing SQL extraction issue...")
    
    os.environ["LOG_LEVEL"] = "INFO"
    orchestrator = DynamicAgentOrchestrator()
    
    # Simple query
    query = "Show me prescribers with Tirosint targeting"
    
    try:
        result = await orchestrator.execute_query(
            query=query,
            user_id="test_user",
            callback=None
        )
        
        print(f"Query completed with status: {result.get('status', 'unknown')}")
        
        # Look for the specific issue
        if '2_query_generation' in result:
            gen_result = result['2_query_generation']
            print(f"Generation result keys: {gen_result.keys()}")
            if 'sql_query' in gen_result:
                sql = gen_result['sql_query']
                print(f"Found SQL query: {sql[:100]}...")
            else:
                print("No sql_query found in generation result")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_sql_extraction())