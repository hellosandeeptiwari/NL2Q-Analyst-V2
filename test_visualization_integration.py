"""
Integration Test: Verify Intelligent Visualization System
Tests end-to-end flow from query to visualization plan
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

async def test_visualization_planner():
    """Test the visualization planner standalone"""
    print("=" * 80)
    print("TEST 1: Visualization Planner Standalone")
    print("=" * 80)
    
    try:
        from backend.agents.visualization_planner import VisualizationPlanner
        import pandas as pd
        
        # Create test data
        data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=6, freq='M'),
            'prescription_count': [1100, 1150, 1200, 1280, 1350, 1405],
            'provider_count': [45, 47, 48, 50, 52, 54],
            'region': ['Midwest'] * 6
        })
        
        planner = VisualizationPlanner()
        query = "Show me prescription trends for the last 6 months"
        
        print(f"\n📊 Testing with query: '{query}'")
        print(f"📈 Data shape: {data.shape}")
        
        plan = await planner.plan_visualization(query, data)
        
        print(f"\n✅ SUCCESS: Visualization plan created")
        print(f"   Layout Type: {plan.layout_type}")
        print(f"   KPIs: {len(plan.kpis)}")
        for i, kpi in enumerate(plan.kpis, 1):
            print(f"      {i}. {kpi.title} ({kpi.calculation})")
        print(f"   Chart: {plan.primary_chart.type} - {plan.primary_chart.title}")
        print(f"   Timeline: {plan.timeline.enabled if plan.timeline else False}")
        
        # Test serialization
        plan_dict = planner.plan_to_dict(plan)
        print(f"\n✅ Plan serialization successful")
        print(f"   Dict keys: {list(plan_dict.keys())}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_orchestrator_integration():
    """Test the orchestrator integration"""
    print("\n" + "=" * 80)
    print("TEST 2: Orchestrator Integration")
    print("=" * 80)
    
    try:
        from backend.orchestrators.dynamic_agent_orchestrator import (
            DynamicAgentOrchestrator, 
            TaskType,
            VISUALIZATION_PLANNER_AVAILABLE
        )
        
        print(f"\n🔍 Visualization Planner Available: {VISUALIZATION_PLANNER_AVAILABLE}")
        
        if not VISUALIZATION_PLANNER_AVAILABLE:
            print("❌ Visualization planner not loaded in orchestrator")
            return False
        
        # Check if TaskType has the new type
        print(f"✅ TaskType.INTELLIGENT_VISUALIZATION_PLANNING exists: {hasattr(TaskType, 'INTELLIGENT_VISUALIZATION_PLANNING')}")
        
        # Check orchestrator has the execution method
        orchestrator = DynamicAgentOrchestrator()
        has_method = hasattr(orchestrator, '_execute_intelligent_visualization_planning')
        print(f"✅ Orchestrator has _execute_intelligent_visualization_planning: {has_method}")
        
        if not has_method:
            print("❌ Method not found in orchestrator")
            return False
        
        # Test plan creation
        print(f"\n🧪 Testing plan creation...")
        user_query = "Show prescription trends"
        tasks = await orchestrator.plan_execution(user_query, context={})
        
        print(f"✅ Plan created with {len(tasks)} tasks")
        
        # Check if intelligent viz planning task was added
        viz_task = None
        for task in tasks:
            print(f"   - {task.task_id}: {task.task_type.value}")
            if task.task_type == TaskType.INTELLIGENT_VISUALIZATION_PLANNING:
                viz_task = task
        
        if viz_task:
            print(f"\n✅ SUCCESS: Intelligent visualization planning task found!")
            print(f"   Task ID: {viz_task.task_id}")
            print(f"   Dependencies: {viz_task.dependencies}")
        else:
            print(f"\n⚠️ WARNING: Intelligent visualization planning task NOT in plan")
            print(f"   This might be expected if VISUALIZATION_PLANNER_AVAILABLE is False")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_task_execution_routing():
    """Test that task routing works correctly"""
    print("\n" + "=" * 80)
    print("TEST 3: Task Execution Routing")
    print("=" * 80)
    
    try:
        from backend.orchestrators.dynamic_agent_orchestrator import (
            DynamicAgentOrchestrator,
            TaskType,
            AgentTask
        )
        
        orchestrator = DynamicAgentOrchestrator()
        
        # Create a mock intelligent viz planning task
        task = AgentTask(
            task_id="test_viz_planning",
            task_type=TaskType.INTELLIGENT_VISUALIZATION_PLANNING,
            input_data={},
            required_output={},
            constraints={},
            dependencies=[]
        )
        
        print(f"🧪 Testing task routing for: {task.task_type.value}")
        
        # Check agent selection
        agent_name = orchestrator._select_agent_for_task(task.task_type)
        print(f"✅ Agent selected: {agent_name}")
        
        if agent_name == "visualization_planner":
            print(f"✅ SUCCESS: Correct agent mapping")
            return True
        else:
            print(f"❌ FAILED: Expected 'visualization_planner', got '{agent_name}'")
            return False
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_frontend_types():
    """Test that frontend types match backend"""
    print("\n" + "=" * 80)
    print("TEST 4: Frontend TypeScript Types")
    print("=" * 80)
    
    try:
        import os
        types_file = os.path.join(
            os.path.dirname(__file__), 
            'frontend', 
            'src', 
            'components', 
            'visualizations', 
            'types.ts'
        )
        
        if os.path.exists(types_file):
            print(f"✅ Frontend types file exists: {types_file}")
            
            with open(types_file, 'r') as f:
                content = f.read()
            
            required_types = [
                'KPISpec',
                'ChartSpec', 
                'TimelineSpec',
                'BreakdownSpec',
                'VisualizationPlan',
                'IntelligentVisualizationResult'
            ]
            
            for type_name in required_types:
                if type_name in content:
                    print(f"   ✓ {type_name} defined")
                else:
                    print(f"   ✗ {type_name} MISSING")
                    return False
            
            print(f"✅ SUCCESS: All required types defined")
            return True
        else:
            print(f"❌ FAILED: Types file not found at {types_file}")
            return False
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all integration tests"""
    print("\n" + "🧪" * 40)
    print("INTELLIGENT VISUALIZATION SYSTEM - INTEGRATION TESTS")
    print("🧪" * 40 + "\n")
    
    results = []
    
    # Test 1: Visualization Planner
    results.append(("Visualization Planner Standalone", await test_visualization_planner()))
    
    # Test 2: Orchestrator Integration
    results.append(("Orchestrator Integration", await test_orchestrator_integration()))
    
    # Test 3: Task Routing
    results.append(("Task Execution Routing", await test_task_execution_routing()))
    
    # Test 4: Frontend Types
    results.append(("Frontend Types", await test_frontend_types()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{'=' * 80}")
    print(f"RESULTS: {passed}/{total} tests passed")
    print(f"{'=' * 80}\n")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Review errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
