#!/usr/bin/env python3

import asyncio
from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator

async def test_orchestrator_direct():
    """Test the orchestrator directly"""
    
    print("🔍 Testing Dynamic Agent Orchestrator directly...")
    
    orchestrator = DynamicAgentOrchestrator()
    
    # Test the exact query that's failing
    result = await orchestrator.process_query(
        user_query="read table final nba output python and fetch top 5 rows and create a visualization with frequency of recommended message and input",
        user_id="default_user",
        session_id="test_session"
    )
    
    print(f"📊 Orchestrator result status: {result.get('status', 'unknown')}")
    print(f"📊 Has plan: {'plan' in result}")
    print(f"📊 Has results: {'results' in result}")
    
    if result.get('status') == 'completed':
        print("✅ Orchestrator completed successfully")
        plan = result.get('plan', {})
        for step_name, step_result in plan.get('step_results', {}).items():
            print(f"  📋 {step_name}: {step_result.get('status', 'unknown')}")
    else:
        print(f"❌ Orchestrator failed: {result.get('error', 'Unknown error')}")
    
    print("✅ Direct orchestrator test completed")

if __name__ == "__main__":
    asyncio.run(test_orchestrator_direct())
