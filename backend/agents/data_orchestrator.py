"""
Enhanced Data Analysis Orchestrator
Combines schema embedding, code generation, and execution
"""
import time
from typing import Dict, List, Any, Optional
from .schema_embedder import SchemaEmbedder
from .code_generator import DataAnalysisCodeGenerator

class DataAnalysisOrchestrator:
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key
        
        # Initialize components
        self.schema_embedder = SchemaEmbedder(openai_api_key)
        self.code_generator = DataAnalysisCodeGenerator(openai_api_key)
        
        self.adapter = None
        self.is_initialized = False
        
        print("🚀 Data Analysis Orchestrator initialized")
    
    def initialize(self, adapter, force_rebuild: bool = False):
        """Initialize with database adapter"""
        self.adapter = adapter
        
        print("🔄 Initializing Data Analysis System...")
        start_time = time.time()
        
        # Try to load cached schemas first
        if not force_rebuild and self.schema_embedder.load_cache():
            print("📁 Using cached schema embeddings")
        else:
            print("🔄 Extracting fresh schema from database...")
            # Extract schema from database
            schemas = self.schema_embedder.extract_schema_from_db(adapter)
            
            if schemas:
                # Create embeddings
                self.schema_embedder.schemas = self.schema_embedder.create_embeddings(schemas)
            else:
                print("⚠️ No schemas extracted")
        
        self.is_initialized = True
        init_time = time.time() - start_time
        
        print(f"✅ Data Analysis System initialized in {init_time:.2f}s")
        print(f"📊 Ready with {len(self.schema_embedder.schemas)} embedded tables")
    
    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        """Complete analysis workflow: find tables, generate code, execute"""
        
        if not self.is_initialized:
            return {"error": "System not initialized"}
        
        start_time = time.time()
        result = {
            "query": user_query,
            "timestamp": start_time,
            "steps": []
        }
        
        try:
            # Step 1: Find relevant tables using embeddings
            print(f"🔍 Finding relevant tables for: '{user_query}'")
            relevant_tables = self.schema_embedder.find_relevant_tables(user_query, top_k=5)
            
            if not relevant_tables:
                return {"error": "No relevant tables found"}
            
            result["relevant_tables"] = [
                {"table": table, "similarity": float(score)} 
                for table, score in relevant_tables
            ]
            result["steps"].append("Found relevant tables using semantic search")
            
            # Step 2: Get schema JSON for code generation
            table_names = [table for table, _ in relevant_tables[:3]]  # Use top 3 tables
            schema_json = self.schema_embedder.get_schema_json(table_names)
            
            result["schema_used"] = schema_json
            result["steps"].append("Retrieved detailed schema for selected tables")
            
            # Step 3: Generate analysis code
            print("🧠 Generating analysis code...")
            code_result = self.code_generator.generate_analysis_code(
                user_query, schema_json, table_names
            )
            
            if "error" in code_result:
                result["error"] = code_result["error"]
                return result
            
            result["generated_code"] = code_result
            result["steps"].append("Generated SQL query and Python analysis code")
            
            # Step 4: Execute SQL and load data
            if code_result.get("sql_query"):
                print("📊 Executing SQL query...")
                data_result = self.code_generator.execute_data_loading(
                    code_result["sql_query"], self.adapter
                )
                
                if "error" in data_result:
                    result["data_loading_error"] = data_result["error"]
                    # Continue without data for code review
                else:
                    result["data_loaded"] = data_result
                    result["steps"].append("Loaded data into pandas DataFrame")
                    
                    # Step 5: Execute analysis code
                    print("🔬 Executing analysis code...")
                    analysis_result = self.code_generator.execute_analysis_code(
                        code_result["code"], data_result.get("dataframe_name")
                    )
                    
                    result["analysis_result"] = analysis_result
                    if "error" not in analysis_result:
                        result["steps"].append("Executed analysis and generated insights")
                    else:
                        result["steps"].append("Analysis execution failed")
            
            # Add timing
            result["processing_time"] = time.time() - start_time
            result["success"] = "error" not in result
            
            return result
            
        except Exception as e:
            result["error"] = f"Analysis failed: {str(e)}"
            result["processing_time"] = time.time() - start_time
            return result
    
    def get_table_suggestions(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Get table suggestions based on semantic similarity"""
        if not self.is_initialized:
            return {"error": "System not initialized"}
        
        relevant_tables = self.schema_embedder.find_relevant_tables(query, top_k)
        
        suggestions = []
        for table_name, similarity in relevant_tables:
            if table_name in self.schema_embedder.schemas:
                schema = self.schema_embedder.schemas[table_name]
                suggestions.append({
                    "table_name": table_name,
                    "similarity_score": float(similarity),
                    "description": schema.description,
                    "column_count": len(schema.columns),
                    "row_count": schema.row_count,
                    "columns": [col["name"] for col in schema.columns[:5]]  # First 5 columns
                })
        
        return {
            "query": query,
            "suggestions": suggestions,
            "total_found": len(suggestions)
        }
    
    def get_schema_for_tables(self, table_names: List[str]) -> Dict[str, Any]:
        """Get detailed schema information for specific tables"""
        return self.schema_embedder.get_schema_json(table_names)
    
    def execute_custom_code(self, code: str, dataframe_name: str = None) -> Dict[str, Any]:
        """Execute custom Python code on stored dataframes"""
        return self.code_generator.execute_analysis_code(code, dataframe_name)
    
    def get_dataframe_info(self, df_name: str = None) -> Dict[str, Any]:
        """Get information about stored dataframes"""
        if df_name:
            return self.code_generator.get_dataframe_info(df_name)
        else:
            return self.code_generator.list_dataframes()
    
    def clear_memory(self):
        """Clear all stored dataframes"""
        self.code_generator.clear_dataframes()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "initialized": self.is_initialized,
            "database_connected": self.adapter is not None,
            "openai_api_available": self.openai_api_key is not None
        }
        
        # Schema embedder status
        if hasattr(self.schema_embedder, 'get_status'):
            status["schema_embedder"] = self.schema_embedder.get_status()
        else:
            status["schema_embedder"] = {
                "tables_embedded": len(self.schema_embedder.schemas)
            }
        
        # Code generator status
        status["code_generator"] = self.code_generator.get_status()
        
        return status
    
    def generate_code_only(self, user_query: str, table_names: List[str] = None) -> Dict[str, Any]:
        """Generate only the code without execution"""
        if not self.is_initialized:
            return {"error": "System not initialized"}
        
        try:
            # Find relevant tables if not provided
            if not table_names:
                relevant_tables = self.schema_embedder.find_relevant_tables(user_query, top_k=3)
                table_names = [table for table, _ in relevant_tables]
            
            if not table_names:
                return {"error": "No relevant tables found"}
            
            # Get schema
            schema_json = self.schema_embedder.get_schema_json(table_names)
            
            # Generate code
            code_result = self.code_generator.generate_analysis_code(
                user_query, schema_json, table_names
            )
            
            return {
                "success": True,
                "query": user_query,
                "tables_used": table_names,
                "schema": schema_json,
                "generated_code": code_result
            }
            
        except Exception as e:
            return {"error": f"Code generation failed: {str(e)}"}
