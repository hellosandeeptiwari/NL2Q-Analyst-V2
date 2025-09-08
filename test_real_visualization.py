#!/usr/bin/env python3
"""
Test the separated visualization workflow with real data
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator

async def test_real_visualization():
    """Test with real data similar to what the server would provide"""
    
    print("🧪 Testing separated visualization with real-like data...")
    
    try:
        # Initialize orchestrator
        orchestrator = DynamicAgentOrchestrator()
        print("✅ Orchestrator initialized")
        
        # Mock data similar to NBA query results
        mock_data = [
            {"Recommended_Msg_Overall": "Increase price", "frequency": 5},
            {"Recommended_Msg_Overall": "Promotional offer", "frequency": 3},
            {"Recommended_Msg_Overall": "Premium bundle", "frequency": 8},
            {"Recommended_Msg_Overall": "Discount campaign", "frequency": 2}
        ]
        
        # Test Python generation
        print("\n🐍 Testing Python generation with real data...")
        
        python_inputs = {
            "original_query": "create a visualization with frequency of recommended message",
            "3_execution": {
                "results": mock_data,
                "sql_query": "SELECT Recommended_Msg_Overall, COUNT(*) as frequency FROM table GROUP BY Recommended_Msg_Overall",
                "status": "success"
            }
        }
        
        python_result = await orchestrator._execute_python_generation(python_inputs)
        
        if python_result.get('status') == 'success':
            python_code = python_result.get('python_code', '')
            print(f"✅ Python generation successful! Generated {len(python_code)} characters of code")
            print(f"📝 First 200 chars: {python_code[:200]}...")
            
            # Test visualization building
            print("\n🎨 Testing visualization building...")
            
            viz_inputs = {
                "original_query": "create a visualization with frequency of recommended message",
                "4_python_generation": python_result
            }
            
            viz_result = await orchestrator._execute_visualization_builder(viz_inputs)
            
            if viz_result.get('status') == 'success':
                charts = viz_result.get('charts', [])
                print(f"✅ Visualization building successful! Generated {len(charts)} charts")
                
                for i, chart in enumerate(charts):
                    chart_type = chart.get('type', 'unknown')
                    title = chart.get('title', 'No title')
                    print(f"  📊 Chart {i+1}: {chart_type} - {title}")
                
                return True
            else:
                print(f"❌ Visualization building failed: {viz_result.get('error')}")
                return False
                
        else:
            print(f"❌ Python generation failed: {python_result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_real_visualization())
    if success:
        print("\n🎉 Separated visualization workflow is working properly!")
    else:
        print("\n💥 There are still issues to resolve")
