import asyncio
import os
from dotenv import load_dotenv
from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator

# Load environment variables
load_dotenv()

async def test_o3_planning():
    orchestrator = DynamicAgentOrchestrator()
    query = 'read table final nba output python and fetch top 5 rows'
    
    print(f"🔍 Testing o3-mini planning for: {query}")
    print(f"🔑 OpenAI API Key available: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    
    try:
        result = await orchestrator.plan_execution(query, 'test_user')
        print('\n=== O3-MINI PLANNING RESULT ===')
        print(f"📊 Result type: {type(result)}")
        
        if isinstance(result, list):
            print(f"📋 Total tasks planned: {len(result)}")
            for i, task in enumerate(result):
                print(f"Task {i+1}: {task}")
        elif isinstance(result, dict):
            for task_id, task_result in result.items():
                print(f'{task_id}: {task_result}')
            print(f"\n📊 Total tasks planned: {len(result)}")
            print(f"📋 Task IDs: {list(result.keys())}")
        else:
            print(f"Unexpected result type: {result}")
            
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_o3_planning())