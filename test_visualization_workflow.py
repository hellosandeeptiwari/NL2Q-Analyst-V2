#!/usr/bin/env python3
"""
Test the fixed visualization workflow
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator

async def test_visualization_methods():
    """Test the visualization generation methods directly"""
    
    print("🔧 Testing visualization workflow methods...")
    
    try:
        # Initialize orchestrator
        orchestrator = DynamicAgentOrchestrator()
        print("✅ Orchestrator initialized")
        
        # Test data similar to what we'd get from NBA query
        test_data = [
            {"Recommended_Msg_Overall": "Message 1", "FREQUENCY": 3},
            {"Recommended_Msg_Overall": "Message 2", "FREQUENCY": 2}
        ]
        
        # Test Python generation
        python_gen_inputs = {
            "original_query": "create visualization with frequency data",
            "3_execution": {
                "data": test_data,
                "sql_query": "SELECT * FROM test"
            }
        }
        
        print("\n🐍 Testing Python generation...")
        python_result = await orchestrator._execute_python_generation(python_gen_inputs)
        
        if python_result.get('status') == 'success':
            print("✅ Python generation successful")
            python_code = python_result.get('python_code', '')
            print(f"📝 Generated {len(python_code)} characters of Python code")
            print("📋 First 200 chars:", python_code[:200] + "..." if len(python_code) > 200 else python_code)
            
            # Test visualization building
            viz_inputs = {
                "original_query": "create visualization with frequency data",
                "4_python_generation": python_result
            }
            
            print("\n🎨 Testing visualization building...")
            viz_result = await orchestrator._execute_visualization_builder(viz_inputs)
            
            if viz_result.get('status') == 'success':
                charts = viz_result.get('charts', [])
                print(f"✅ Visualization building successful - {len(charts)} charts generated")
                if charts:
                    for i, chart in enumerate(charts):
                        print(f"   📊 Chart {i+1}: {chart.get('type', 'unknown')} - {chart.get('title', 'no title')}")
                else:
                    print("⚠️ No charts generated, but no error occurred")
            else:
                print(f"❌ Visualization building failed: {viz_result.get('error')}")
                
        else:
            print(f"❌ Python generation failed: {python_result.get('error')}")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_visualization_methods())
    if success:
        print("\n✨ Visualization workflow test completed!")
    else:
        print("\n💥 Test failed - check implementation")
