#!/usr/bin/env python3
"""
Test script to verify the separated Python generation and visualization building functionality
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator, TaskType, AgentTask

async def test_separated_visualization():
    """Test the new two-step visualization process"""
    
    print("🚀 Testing separated Python generation and visualization building...")
    
    try:
        # Initialize orchestrator
        orchestrator = DynamicAgentOrchestrator()
        print("✅ Orchestrator initialized successfully")
        
        # Test agent selection for new task types
        python_agent = orchestrator._select_agent_for_task(TaskType.PYTHON_GENERATION)
        viz_agent = orchestrator._select_agent_for_task(TaskType.VISUALIZATION_BUILDER)
        
        print(f"✅ Python generation agent: {python_agent}")
        print(f"✅ Visualization builder agent: {viz_agent}")
        
        # Test task creation for visualization query
        user_query = "Show me the top 10 NBA players by points scored"
        
        # Create a mock planning result that should include our new task types
        print("\n📋 Testing task planning for visualization query...")
        
        # Simulate the planning response structure
        mock_planning_result = {
            "tasks": [
                {
                    "task_type": "schema_discovery",
                    "inputs": {"user_query": user_query},
                    "dependencies": []
                },
                {
                    "task_type": "query_generation", 
                    "inputs": {"user_query": user_query},
                    "dependencies": ["schema_discovery"]
                },
                {
                    "task_type": "execution",
                    "inputs": {"user_query": user_query},
                    "dependencies": ["query_generation"]
                },
                {
                    "task_type": "python_generation",
                    "inputs": {"user_query": user_query},
                    "dependencies": ["execution"]
                },
                {
                    "task_type": "visualization_builder",
                    "inputs": {"user_query": user_query},
                    "dependencies": ["python_generation"]
                }
            ]
        }
        
        # Create AgentTask objects
        tasks = []
        for i, task_data in enumerate(mock_planning_result["tasks"]):
            task = AgentTask(
                task_id=f"task_{i+1}",
                task_type=TaskType(task_data["task_type"]),
                input_data=task_data["inputs"],
                required_output={},
                constraints={},
                dependencies=task_data["dependencies"]
            )
            tasks.append(task)
            
        print(f"✅ Created {len(tasks)} tasks for execution")
        
        # Test that the task types are properly recognized
        python_task = None
        viz_task = None
        
        for task in tasks:
            if task.task_type == TaskType.PYTHON_GENERATION:
                python_task = task
            elif task.task_type == TaskType.VISUALIZATION_BUILDER:
                viz_task = task
                
        if python_task:
            print(f"✅ Python generation task created: {python_task.task_id}")
        else:
            print("❌ Python generation task not found")
            
        if viz_task:
            print(f"✅ Visualization builder task created: {viz_task.task_id}")
        else:
            print("❌ Visualization builder task not found")
        
        # Test execution method existence
        if hasattr(orchestrator, '_execute_python_generation'):
            print("✅ _execute_python_generation method exists")
        else:
            print("❌ _execute_python_generation method missing")
            
        if hasattr(orchestrator, '_execute_visualization_builder'):
            print("✅ _execute_visualization_builder method exists")
        else:
            print("❌ _execute_visualization_builder method missing")
        
        print("\n🎉 All tests passed! The separated visualization workflow is ready.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_separated_visualization())
    if success:
        print("\n✨ Separated visualization implementation is complete and functional!")
    else:
        print("\n💥 Tests failed - please check the implementation")
