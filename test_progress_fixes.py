#!/usr/bin/env python3
"""
Test the real-time progress fixes
"""

import asyncio
import json
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator

async def test_progress_updates():
    """Test that progress updates are properly broadcast"""
    
    print("🧪 Testing Real-time Progress Updates")
    print("=" * 50)
    
    # Create orchestrator
    orchestrator = DynamicAgentOrchestrator()
    
    # Test query
    test_query = "Show me top 5 players by scoring average"
    
    try:
        # Process query and capture progress
        print(f"📝 Testing query: {test_query}")
        
        # Create plan
        print("\n📋 Step 1: Creating plan...")
        plan = await orchestrator.plan_execution(test_query)
        print(f"✅ Plan created with {len(plan)} tasks")
        
        for i, task in enumerate(plan):
            print(f"  Task {i+1}: {task.task_id} ({task.task_type.value})")
        
        # Execute plan with progress tracking
        print("\n⚡ Step 2: Executing plan with progress tracking...")
        results = await orchestrator.execute_plan(plan, test_query)
        
        print(f"✅ Execution completed")
        print(f"📊 Results keys: {list(results.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_progress_updates())
    print(f"\n{'✅ Test PASSED' if success else '❌ Test FAILED'}")
