#!/usr/bin/env python3
"""
Enhanced NBA Data Analysis Server
Complete workflow: Schema embedding -> Code generation -> Data analysis
"""
import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Initialize database
from db.engine import get_adapter
adapter = get_adapter()

# Initialize Data Analysis Orchestrator
from agents.data_orchestrator import DataAnalysisOrchestrator
data_orchestrator = DataAnalysisOrchestrator(openai_api_key=os.getenv('OPENAI_API_KEY'))

# Initialize FastAPI
app = FastAPI(
    title="NBA Data Analysis API",
    description="Enhanced API with schema embedding and code generation",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class AnalysisRequest(BaseModel):
    query: str
    force_rebuild: Optional[bool] = False

class CodeRequest(BaseModel):
    query: str
    tables: Optional[List[str]] = None

class ExecuteCodeRequest(BaseModel):
    code: str
    dataframe_name: Optional[str] = None

# Initialize the system
def initialize_system():
    """Initialize the data analysis system"""
    try:
        print("🚀 Initializing Enhanced NBA Data Analysis System...")
        data_orchestrator.initialize(adapter)
        print("✅ System initialization complete")
    except Exception as e:
        print(f"⚠️ System initialization failed: {e}")

# Health endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        db_health = adapter.health()
        system_status = data_orchestrator.get_system_status()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": db_health,
            "system": system_status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# System status
@app.get("/status")
async def get_system_status():
    """Get detailed system status"""
    return data_orchestrator.get_system_status()

# Table suggestions
@app.get("/suggest-tables")
async def suggest_tables(query: str, top_k: int = 10):
    """Get table suggestions based on semantic similarity"""
    try:
        suggestions = data_orchestrator.get_table_suggestions(query, top_k)
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Table suggestion failed: {str(e)}")

# Get schema for specific tables
@app.post("/schema")
async def get_schema(table_names: List[str]):
    """Get detailed schema for specific tables"""
    try:
        schema = data_orchestrator.get_schema_for_tables(table_names)
        return schema
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schema retrieval failed: {str(e)}")

# Generate code only
@app.post("/generate-code")
async def generate_code(request: CodeRequest):
    """Generate analysis code without execution"""
    try:
        result = data_orchestrator.generate_code_only(request.query, request.tables)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

# Full analysis workflow
@app.post("/analyze")
async def analyze_data(request: AnalysisRequest):
    """Complete analysis workflow: find tables, generate code, execute"""
    try:
        result = data_orchestrator.analyze_query(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Execute custom code
@app.post("/execute")
async def execute_code(request: ExecuteCodeRequest):
    """Execute custom Python code on stored dataframes"""
    try:
        result = data_orchestrator.execute_custom_code(request.code, request.dataframe_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code execution failed: {str(e)}")

# Dataframe management
@app.get("/dataframes")
async def list_dataframes():
    """List all stored dataframes"""
    try:
        return data_orchestrator.get_dataframe_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list dataframes: {str(e)}")

@app.get("/dataframes/{df_name}")
async def get_dataframe_info(df_name: str):
    """Get information about a specific dataframe"""
    try:
        return data_orchestrator.get_dataframe_info(df_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataframe info: {str(e)}")

@app.delete("/dataframes")
async def clear_dataframes():
    """Clear all stored dataframes"""
    try:
        data_orchestrator.clear_memory()
        return {"message": "All dataframes cleared from memory"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear dataframes: {str(e)}")

# Test endpoints
@app.get("/test-nba")
async def test_nba_query():
    """Test NBA-specific query"""
    test_query = "Show me NBA player statistics with points, rebounds, and assists"
    try:
        result = data_orchestrator.analyze_query(test_query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test query failed: {str(e)}")

@app.get("/test-embedding")
async def test_embedding():
    """Test schema embedding functionality"""
    try:
        test_queries = [
            "NBA basketball player data",
            "Customer purchase information",
            "Player scoring statistics"
        ]
        
        results = {}
        for query in test_queries:
            suggestions = data_orchestrator.get_table_suggestions(query, top_k=3)
            results[query] = suggestions
        
        return {
            "test_queries": results,
            "system_status": data_orchestrator.get_system_status()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding test failed: {str(e)}")

# Initialize system on startup
@app.on_event("startup")
async def startup_event():
    initialize_system()

def run_server():
    """Run the enhanced server"""
    import uvicorn
    
    port = int(os.getenv("PORT", 8004))
    
    print(f"\n🚀 Enhanced NBA Data Analysis Server")
    print(f"📊 Schema Embedding: OpenAI text-embedding-3-small")
    print(f"🧠 Code Generation: GPT-3.5-turbo")
    print(f"🐍 Execution Environment: Pandas + Matplotlib")
    print(f"🌐 Server running on http://localhost:{port}")
    print(f"\n📋 Available Endpoints:")
    print(f"   GET  /health - System health check")
    print(f"   GET  /status - Detailed system status")
    print(f"   GET  /suggest-tables?query=... - Get table suggestions")
    print(f"   POST /schema - Get schema for specific tables")
    print(f"   POST /generate-code - Generate analysis code")
    print(f"   POST /analyze - Complete analysis workflow")
    print(f"   POST /execute - Execute custom Python code")
    print(f"   GET  /dataframes - List stored dataframes")
    print(f"   GET  /test-nba - Test NBA query")
    print(f"   GET  /test-embedding - Test embedding functionality")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except KeyboardInterrupt:
        print("\n👋 Server shutdown requested")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    run_server()
