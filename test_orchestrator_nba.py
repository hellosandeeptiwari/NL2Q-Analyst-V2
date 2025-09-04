#!/usr/bin/env python3
"""
Test Dynamic Orchestrator with NBA Query
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

async def test_orchestrator_nba():
    """Test orchestrator with NBA query"""
    
    print("🎬 Testing Dynamic Orchestrator with NBA Query...")
    
    try:
        from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
        
        # Create orchestrator
        orchestrator = DynamicAgentOrchestrator()
        
        # Test query
        query = "read table final nba output python and fetch top 5 rows and create a visualization with frequency of recommended message and provider input"
        
        print(f"🔍 Query: {query}")
        print("="*80)
        
        # Plan execution
        tasks = await orchestrator.plan_execution(query)
        print(f"📋 Planned {len(tasks)} tasks")
        
        # Execute plan
        result = await orchestrator.execute_plan(tasks, query)
        
        print(f"\n📋 ORCHESTRATION RESULT:")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Total tasks: {result.get('total_tasks', 0)}")
        print(f"   Completed: {result.get('completed_tasks', 0)}")
        
        if result.get('task_results'):
            print(f"\n📊 TASK RESULTS:")
            for task_id, task_result in result['task_results'].items():
                print(f"   {task_id}:")
                if 'discovered_tables' in task_result:
                    tables = task_result['discovered_tables']
                    print(f"      - Discovered tables: {len(tables)} tables")
                    for table in tables[:3]:  # Show first 3
                        print(f"        • {table}")
                elif 'matched_tables' in task_result:
                    tables = task_result['matched_tables']
                    print(f"      - Matched tables: {len(tables)} tables")
                    for table in tables:
                        print(f"        • {table}")
                elif 'entities' in task_result:
                    entities = task_result['entities']
                    print(f"      - Entities: {entities}")
                elif 'sql_query' in task_result:
                    sql = task_result['sql_query']
                    print(f"      - SQL: {sql[:100]}...")
                elif 'visualization_code' in task_result:
                    print(f"      - Visualization: Generated")
                else:
                    status = task_result.get('status', 'unknown')
                    print(f"      - Status: {status}")
                    if 'error' in task_result:
                        print(f"      - Error: {task_result['error']}")
        
        if result.get('final_result'):
            print(f"\n🎯 FINAL RESULT:")
            final = result['final_result']
            print(f"   Type: {type(final)}")
            if isinstance(final, dict):
                for key, value in final.items():
                    if key == 'visualization_code':
                        print(f"   {key}: Generated ({len(str(value))} chars)")
                    else:
                        print(f"   {key}: {str(value)[:100]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(test_orchestrator_nba())
    
    if result and result.get('success'):
        print(f"\n🎉 ORCHESTRATOR TEST SUCCESSFUL!")
        print(f"✅ Dynamic NL2Q pipeline working end-to-end")
    else:
        print(f"\n❌ Orchestrator test failed")
