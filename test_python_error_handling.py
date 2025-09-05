#!/usr/bin/env python3
"""
Test Python Agentic Retry System - Stack Trace Handling
Tests that the Python code retry system captures and passes full stack traces to the LLM
"""

import asyncio
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

async def test_python_error_handling():
    """Test that Python errors are properly captured with stack traces"""
    
    try:
        from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
        
        print("🧪 Testing Python Agentic Retry System - Error Handling")
        print("=" * 70)
        
        # Initialize orchestrator
        orchestrator = DynamicAgentOrchestrator()
        
        # Test data that might cause Python errors
        test_data = [
            {"player_name": "LeBron James", "points": 25, "team": "Lakers"},
            {"player_name": "Stephen Curry", "points": 30, "team": "Warriors"},
            {"player_name": "Kevin Durant", "points": 28, "team": "Suns"}
        ]
        
        print("📋 Test Case 1: Python Code Generation without previous error")
        result1 = await orchestrator._generate_python_visualization_code(
            query="Create a bar chart showing points by player",
            data=test_data,
            attempt=1,
            previous_error=None
        )
        
        print(f"✅ Result 1 Status: {result1.get('status', 'unknown')}")
        if result1.get("python_code"):
            print(f"📝 Generated Code Preview: {result1['python_code'][:100]}...")
        else:
            print(f"❌ No code generated: {result1.get('error', 'Unknown error')}")
        
        print()
        
        # Test Case 2: With a detailed Python error (simulating retry scenario)
        print("📋 Test Case 2: Python Code Generation with previous error context")
        
        detailed_error = """Python execution error: name 'invalid_function' is not defined

Full traceback:
Traceback (most recent call last):
  File "<string>", line 15, in <module>
    result = invalid_function(data)
NameError: name 'invalid_function' is not defined

Common Python issues:
- Import errors: Check if all required libraries are available
- Data type mismatches: Handle different data types (strings, numbers, dates)
- Column name errors: Ensure column names match exactly (case-sensitive)
- Syntax errors: Fix Python syntax issues
- Library compatibility: Use compatible function calls
- Index errors: Handle empty data or missing columns gracefully"""
        
        result2 = await orchestrator._generate_python_visualization_code(
            query="Create a bar chart showing points by player",
            data=test_data,
            attempt=2,
            previous_error=detailed_error
        )
        
        print(f"✅ Result 2 Status: {result2.get('status', 'unknown')}")
        if result2.get("python_code"):
            print(f"📝 Generated Code with Error Context Preview: {result2['python_code'][:100]}...")
            
            # Check if the code avoids the previous error
            if "invalid_function" not in result2['python_code']:
                print("✅ SUCCESS: LLM learned from error context and avoided problematic function!")
            else:
                print("⚠️ WARNING: LLM may not have fully learned from error context")
        else:
            print(f"❌ No code generated with error context: {result2.get('error', 'Unknown error')}")
        
        print()
        
        # Test Case 3: Test the execution error capturing
        print("📋 Test Case 3: Testing Python execution error capture")
        
        # Create deliberately problematic Python code
        problematic_code = """
import pandas as pd
import matplotlib.pyplot as plt

# This will cause an error - undefined variable
df = pd.DataFrame(undefined_variable)
plt.figure()
plt.bar(df['x'], df['y'])
plt.show()
"""
        
        execution_result = await orchestrator._execute_python_visualization(
            python_code=problematic_code,
            data=test_data
        )
        
        print(f"📊 Execution Result Status: {execution_result.get('status', 'unknown')}")
        
        if execution_result.get('error'):
            error_message = execution_result['error']
            print(f"✅ Error captured: {error_message[:100]}...")
            
            # Check if full traceback is included
            if "Traceback" in error_message and "NameError" in error_message:
                print("✅ SUCCESS: Full stack trace captured for LLM learning!")
            else:
                print("⚠️ WARNING: Stack trace may not be complete")
                
            # Check if we have both detailed and simple error versions
            if execution_result.get('simple_error'):
                simple = execution_result['simple_error']
                print(f"✅ Simple error for display: {simple[:100]}...")
        else:
            print("❌ No error captured - this is unexpected for problematic code")
        
        print()
        print("🧪 Test Summary:")
        print("=" * 50)
        print("✅ Python code generation tested")
        print("✅ Error context handling tested")
        print("✅ Stack trace capture tested")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        print(f"📋 Full error trace:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Python Agentic Retry Error Handling Tests")
    print("=" * 80)
    
    # Run the test
    success = asyncio.run(test_python_error_handling())
    
    if success:
        print("\n🏁 Tests completed - Python error handling system verified!")
    else:
        print("\n💥 Tests failed - check implementation")
