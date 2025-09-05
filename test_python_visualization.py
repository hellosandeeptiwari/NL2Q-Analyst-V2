#!/usr/bin/env python3

import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import io
import base64
from contextlib import redirect_stdout, redirect_stderr

async def test_python_visualization():
    """Test Python visualization execution with fixed imports"""
    
    print("🔍 Testing Python visualization execution...")
    
    # Sample data similar to what we get from NBA table
    data = [
        {'input_id': 1, 'Expected_Value_avg': 85.2, 'Recommended_Msg_Overall': 'Increase engagement', 'Action_rank': 1},
        {'input_id': 2, 'Expected_Value_avg': 92.1, 'Recommended_Msg_Overall': 'Target premium users', 'Action_rank': 2},
        {'input_id': 3, 'Expected_Value_avg': 78.5, 'Recommended_Msg_Overall': 'Increase engagement', 'Action_rank': 3},
        {'input_id': 4, 'Expected_Value_avg': 88.9, 'Recommended_Msg_Overall': 'Personalize content', 'Action_rank': 4},
        {'input_id': 5, 'Expected_Value_avg': 81.3, 'Recommended_Msg_Overall': 'Increase engagement', 'Action_rank': 5}
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Test Python code that should work with our fixed environment
    python_code = """
# Create a bar chart showing frequency of recommended messages
msg_counts = df['Recommended_Msg_Overall'].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(msg_counts.index, msg_counts.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Frequency of Recommended Messages')
plt.xlabel('Recommended Message')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
"""
    
    # Prepare safe execution environment (same as in orchestrator)
    safe_globals = {
        'pd': pd,
        'df': df,
        'plt': plt,
        'px': px,
        'go': go,
        'sns': sns,
        'np': np,
        'data': data,
        '__import__': __import__,  # Allow imports at top level
        '__builtins__': {
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'bool': bool,
            'print': print,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'min': min,
            'max': max,
            'sum': sum,
            'abs': abs,
            'round': round,
            '__import__': __import__,  # Also keep in builtins
            'type': type,
            'isinstance': isinstance,
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
            'sorted': sorted,
            'any': any,
            'all': all
        }
    }
    
    safe_locals = {}
    
    # Capture output
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(python_code, safe_globals, safe_locals)
        
        # Check for matplotlib figures
        if plt.get_fignums():
            print(f"✅ Python code executed successfully! Created {len(plt.get_fignums())} figure(s)")
            
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                
                # Convert to base64
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                print(f"📊 Chart created successfully - base64 length: {len(img_base64)}")
                plt.close(fig)
        else:
            print("⚠️ No matplotlib figures were created")
            
        stdout_output = stdout_buffer.getvalue()
        stderr_output = stderr_buffer.getvalue()
        
        if stdout_output:
            print(f"📝 Stdout: {stdout_output}")
        if stderr_output:
            print(f"⚠️ Stderr: {stderr_output}")
            
    except Exception as e:
        print(f"❌ Python execution failed: {str(e)}")
        stderr_output = stderr_buffer.getvalue()
        if stderr_output:
            print(f"Error details: {stderr_output}")
    
    print("✅ Visualization test completed")

if __name__ == "__main__":
    asyncio.run(test_python_visualization())
