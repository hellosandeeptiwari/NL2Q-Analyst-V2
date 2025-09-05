#!/usr/bin/env python3

"""
Test Python Agentic Retry System
Tests the new agentic Python code generation and execution with retry mechanism
"""

import asyncio
import pandas as pd
from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator

async def test_python_agentic_retry():
    """Test the Python agentic retry system"""
    print("🧪 Testing Python Agentic Retry System")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = DynamicAgentOrchestrator()
    
    # Sample data for testing
    test_data = [
        {"player_name": "LeBron James", "points": 25.0, "assists": 7.3, "rebounds": 7.4, "team": "Lakers"},
        {"player_name": "Stephen Curry", "points": 29.5, "assists": 6.1, "rebounds": 5.2, "team": "Warriors"},
        {"player_name": "Kevin Durant", "points": 29.7, "assists": 5.0, "rebounds": 7.1, "team": "Suns"},
        {"player_name": "Giannis Antetokounmpo", "points": 31.1, "assists": 5.7, "rebounds": 11.8, "team": "Bucks"},
        {"player_name": "Luka Doncic", "points": 32.4, "assists": 8.0, "rebounds": 8.6, "team": "Mavericks"}
    ]
    
    # Test cases
    test_cases = [
        {
            "name": "Simple Visualization Query",
            "query": "Create a bar chart showing player points",
            "should_use_python": False
        },
        {
            "name": "Advanced Python Visualization Query",
            "query": "Create a complex statistical analysis with correlation heatmap and regression analysis between points, assists, and rebounds using Python",
            "should_use_python": True
        },
        {
            "name": "Advanced Plotly Query",
            "query": "Generate advanced plotly visualization with interactive features for player performance analysis",
            "should_use_python": True
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🔍 Test Case: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        print(f"Expected Python usage: {test_case['should_use_python']}")
        
        # Test _requires_python_visualization method
        requires_python = orchestrator._requires_python_visualization(test_case['query'], test_data)
        print(f"System decision: {'Python' if requires_python else 'Standard'} visualization")
        
        # Validate decision
        if requires_python == test_case['should_use_python']:
            print("✅ Correct decision")
        else:
            print("❌ Incorrect decision")
        
        # Test visualization execution
        try:
            inputs = {
                "original_query": test_case['query'],
                "6_query_execution": {
                    "results": test_data
                }
            }
            
            print("🚀 Testing visualization execution...")
            result = await orchestrator._execute_visualization(inputs)
            
            if result.get("status") == "completed":
                print("✅ Visualization completed successfully")
                print(f"Charts generated: {len(result.get('charts', []))}")
                print(f"Chart types: {result.get('chart_types', [])}")
                
                if 'generation_attempts' in result:
                    print(f"Python generation attempts: {result['generation_attempts']}")
                if 'python_code' in result:
                    print("✅ Python code was generated")
                if 'fallback_used' in result:
                    print(f"Fallback used: {result['fallback_used']}")
            else:
                print(f"❌ Visualization failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
        
        print("-" * 30)

async def test_python_code_generation():
    """Test Python code generation specifically"""
    print("\n🐍 Testing Python Code Generation")
    print("=" * 50)
    
    orchestrator = DynamicAgentOrchestrator()
    
    test_data = [
        {"category": "A", "value": 10},
        {"category": "B", "value": 20},
        {"category": "C", "value": 15}
    ]
    
    try:
        print("🔧 Testing Python code generation...")
        result = await orchestrator._generate_python_visualization_code(
            query="Create a matplotlib bar chart with custom colors and styling",
            data=test_data,
            attempt=1
        )
        
        if result.get("status") == "success":
            print("✅ Python code generation successful")
            print("Generated code preview:")
            code = result.get("python_code", "")
            print("```python")
            print(code[:300] + "..." if len(code) > 300 else code)
            print("```")
        else:
            print(f"❌ Python code generation failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Python code generation test failed: {e}")

async def main():
    """Run all tests"""
    print("🚀 Starting Python Agentic Retry System Tests")
    print("=" * 60)
    
    await test_python_agentic_retry()
    await test_python_code_generation()
    
    print("\n" + "=" * 60)
    print("🏁 Tests completed")

if __name__ == "__main__":
    asyncio.run(main())
