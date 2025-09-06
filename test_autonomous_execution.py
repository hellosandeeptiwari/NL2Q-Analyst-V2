import asyncio
import os
from dotenv import load_dotenv
from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator

# Load environment variables
load_dotenv()

async def test_autonomous_execution():
    orchestrator = DynamicAgentOrchestrator()
    query = 'read table final nba output python and fetch top 5 rows'
    
    print(f"🔍 Testing truly autonomous execution for: {query}")
    print(f"🔑 OpenAI API Key available: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    
    try:
        # First, see what o3-mini plans
        result = await orchestrator.plan_execution(query, 'test_user')
        print('\n=== O3-MINI PLANNING ===')
        print(f"📊 Planned {len(result)} tasks:")
        for i, task in enumerate(result):
            print(f"  {i+1}. {task.task_id}: {task.task_type.value}")
        
        # Now execute the plan
        print('\n=== EXECUTING PLAN ===')
        execution_result = await orchestrator.execute_plan(result, query, 'test_user')
        
        print('\n=== EXECUTION RESULTS ===')
        for task_id, task_result in execution_result.items():
            print(f"\n{task_id}:")
            if isinstance(task_result, dict):
                status = task_result.get('status', 'unknown')
                print(f"  Status: {status}")
                if 'error' in task_result:
                    print(f"  Error: {task_result['error']}")
                elif 'sql_query' in task_result:
                    print(f"  SQL: {task_result['sql_query']}")
                elif 'results' in task_result:
                    results = task_result['results']
                    print(f"  Results: {len(results) if results else 0} rows")
                    if results and len(results) > 0:
                        print(f"  Sample: {str(results[0])[:100]}...")
            else:
                print(f"  Result: {str(task_result)[:100]}...")
            
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_autonomous_execution())
