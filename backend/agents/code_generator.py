"""
Python Code Generator for Data Analysis
Generates executable Python code based on schema and user queries
"""
import json
import os
import openai
from typing import Dict, List, Any, Optional
import pandas as pd
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

class DataAnalysisCodeGenerator:
    def __init__(self, api_key: str = None):
        self.api_key = os.getenv('OPENAI_API_KEY') or api_key
        if self.api_key:
            openai.api_key = self.api_key
            print("✅ OpenAI API key loaded for code generation")
        else:
            print("⚠️ No OpenAI API key - code generation unavailable")
        
        self.model = "gpt-3.5-turbo"
        self.dataframes: Dict[str, pd.DataFrame] = {}
        
    def generate_analysis_code(self, user_query: str, schema_json: Dict, 
                             relevant_tables: List[str]) -> Dict[str, Any]:
        """Generate Python code for data analysis based on user query and schema"""
        
        if not self.api_key:
            return {"error": "OpenAI API not available"}
        
        try:
            # Create focused schema for relevant tables
            focused_schema = {
                "tables": {table: schema_json["tables"][table] 
                          for table in relevant_tables if table in schema_json["tables"]}
            }
            
            # Create code generation prompt
            prompt = self._create_code_prompt(user_query, focused_schema)
            
            # Generate code using OpenAI
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            generated_code = response.choices[0].message.content
            
            # Parse the response to extract code and explanation
            code_parts = self._parse_generated_response(generated_code)
            
            return {
                "success": True,
                "code": code_parts["code"],
                "explanation": code_parts["explanation"],
                "sql_query": code_parts.get("sql_query", ""),
                "analysis_steps": code_parts.get("steps", [])
            }
            
        except Exception as e:
            return {"error": f"Code generation failed: {e}"}
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for code generation"""
        return """You are an expert Python data analyst. Generate clean, executable Python code for data analysis tasks.

REQUIREMENTS:
1. Generate SQL query to fetch data from the database
2. Load data into pandas DataFrame
3. Perform requested analysis (plotting, insights, statistics)
4. Use matplotlib/seaborn for visualizations
5. Include data validation and error handling
6. Add clear comments explaining each step

RESPONSE FORMAT:
```sql
-- SQL query to fetch data
SELECT ... FROM ...
```

```python
# Python code for analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load data (assume data is already loaded from SQL)
# df = pd.read_sql(sql_query, connection)

# Step 2: Data exploration and cleaning
# ... analysis code ...

# Step 3: Generate insights and visualizations
# ... plotting code ...
```

EXPLANATION:
Brief explanation of what the code does and insights it provides.

STEPS:
1. Data loading from specified tables
2. Data cleaning and preprocessing  
3. Analysis and visualization
4. Key insights generation"""

    def _create_code_prompt(self, user_query: str, schema_json: Dict) -> str:
        """Create prompt for code generation"""
        prompt_parts = []
        
        prompt_parts.append(f"USER REQUEST: {user_query}")
        prompt_parts.append("\nAVAILABLE TABLES AND COLUMNS:")
        
        for table_name, table_info in schema_json["tables"].items():
            prompt_parts.append(f"\nTable: {table_name}")
            prompt_parts.append(f"Description: {table_info.get('description', 'No description')}")
            prompt_parts.append(f"Rows: {table_info.get('row_count', 'Unknown')}")
            prompt_parts.append("Columns:")
            
            for col in table_info["columns"]:
                col_desc = f"  - {col['name']} ({col['type']})"
                if col.get('nullable') == 'NO':
                    col_desc += " [NOT NULL]"
                prompt_parts.append(col_desc)
        
        prompt_parts.append("\nGenerate SQL query and Python analysis code for this request.")
        
        return "\n".join(prompt_parts)
    
    def _parse_generated_response(self, response: str) -> Dict[str, Any]:
        """Parse the generated response to extract components"""
        code_parts = {
            "code": "",
            "explanation": "",
            "sql_query": "",
            "steps": []
        }
        
        lines = response.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            if line.strip().startswith('```sql'):
                current_section = 'sql'
                current_content = []
            elif line.strip().startswith('```python'):
                current_section = 'python'
                current_content = []
            elif line.strip() == '```':
                if current_section == 'sql':
                    code_parts["sql_query"] = '\n'.join(current_content)
                elif current_section == 'python':
                    code_parts["code"] = '\n'.join(current_content)
                current_section = None
                current_content = []
            elif line.strip().startswith('EXPLANATION:'):
                current_section = 'explanation'
                current_content = []
            elif line.strip().startswith('STEPS:'):
                current_section = 'steps'
                current_content = []
            elif current_section:
                if current_section == 'explanation':
                    current_content.append(line)
                elif current_section == 'steps':
                    if line.strip() and not line.startswith(' '):
                        current_content.append(line.strip())
                else:
                    current_content.append(line)
        
        # Handle remaining content
        if current_section == 'explanation':
            code_parts["explanation"] = '\n'.join(current_content).strip()
        elif current_section == 'steps':
            code_parts["steps"] = [step for step in current_content if step.strip()]
        
        # If no structured response, try to extract code blocks
        if not code_parts["code"] and '```' in response:
            import re
            code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', response, re.DOTALL)
            if code_blocks:
                code_parts["code"] = code_blocks[-1]  # Take the last code block
        
        return code_parts
    
    def execute_data_loading(self, sql_query: str, adapter) -> Dict[str, Any]:
        """Execute SQL query and load data into pandas DataFrame"""
        try:
            # Execute SQL query
            result = adapter.run(sql_query)
            
            if result.error:
                return {"error": f"SQL execution failed: {result.error}"}
            
            # Convert to pandas DataFrame
            if result.rows:
                # Get column names (this might need adjustment based on your adapter)
                # For now, we'll use generic column names
                columns = [f"col_{i}" for i in range(len(result.rows[0]))]
                df = pd.DataFrame(result.rows, columns=columns)
                
                # Store in memory
                df_name = f"data_{len(self.dataframes)}"
                self.dataframes[df_name] = df
                
                return {
                    "success": True,
                    "dataframe_name": df_name,
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "sample_data": df.head().to_dict() if not df.empty else {}
                }
            else:
                return {"error": "No data returned from query"}
                
        except Exception as e:
            return {"error": f"Data loading failed: {e}"}
    
    def execute_analysis_code(self, code: str, dataframe_name: str = None) -> Dict[str, Any]:
        """Execute the generated Python analysis code"""
        try:
            # Prepare execution environment
            exec_globals = {
                'pd': pd,
                'plt': None,
                'sns': None,
                'np': None,
                'dataframes': self.dataframes
            }
            
            # Import common libraries
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                import numpy as np
                exec_globals.update({'plt': plt, 'sns': sns, 'np': np})
            except ImportError as e:
                print(f"⚠️ Some libraries not available: {e}")
            
            # Add the main dataframe to globals if specified
            if dataframe_name and dataframe_name in self.dataframes:
                exec_globals['df'] = self.dataframes[dataframe_name]
            
            # Capture output
            output_buffer = io.StringIO()
            error_buffer = io.StringIO()
            
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                exec(code, exec_globals)
            
            output = output_buffer.getvalue()
            errors = error_buffer.getvalue()
            
            # Get any new dataframes created
            new_dataframes = {}
            for key, value in exec_globals.items():
                if isinstance(value, pd.DataFrame) and key not in ['pd']:
                    new_dataframes[key] = {
                        "shape": value.shape,
                        "columns": list(value.columns),
                        "sample": value.head().to_dict() if not value.empty else {}
                    }
                    # Store in memory
                    self.dataframes[key] = value
            
            result = {
                "success": True,
                "output": output,
                "new_dataframes": new_dataframes,
                "total_dataframes": len(self.dataframes)
            }
            
            if errors:
                result["warnings"] = errors
                
            return result
            
        except Exception as e:
            return {"error": f"Code execution failed: {str(e)}"}
    
    def get_dataframe_info(self, df_name: str) -> Dict[str, Any]:
        """Get information about a stored dataframe"""
        if df_name not in self.dataframes:
            return {"error": f"DataFrame '{df_name}' not found"}
        
        df = self.dataframes[df_name]
        
        return {
            "name": df_name,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage().sum(),
            "sample_data": df.head().to_dict(),
            "description": df.describe().to_dict() if not df.empty else {}
        }
    
    def list_dataframes(self) -> Dict[str, Any]:
        """List all stored dataframes"""
        df_info = {}
        for name, df in self.dataframes.items():
            df_info[name] = {
                "shape": df.shape,
                "columns": list(df.columns),
                "memory_mb": round(df.memory_usage().sum() / 1024 / 1024, 2)
            }
        
        return {
            "total_dataframes": len(self.dataframes),
            "dataframes": df_info
        }
    
    def clear_dataframes(self):
        """Clear all stored dataframes"""
        self.dataframes.clear()
        print("🗑️ Cleared all dataframes from memory")
    
    def get_status(self) -> Dict[str, Any]:
        """Get code generator status"""
        return {
            "api_available": self.api_key is not None,
            "model": self.model,
            "dataframes_in_memory": len(self.dataframes),
            "total_memory_mb": sum(df.memory_usage().sum() for df in self.dataframes.values()) / 1024 / 1024
        }
