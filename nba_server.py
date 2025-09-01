#!/usr/bin/env python3
"""
Working NBA query server
"""
import sys
from pathlib import Path
import json

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Create app
app = FastAPI(title="NL2Q Agent", version="1.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
from db.engine import get_adapter
adapter = get_adapter()

class QueryRequest(BaseModel):
    natural_language: str
    job_id: str = "default"

@app.get("/")
def root():
    return {"status": "NBA Query Agent Ready", "database": "Snowflake"}

@app.get("/health")
def health():
    return adapter.health()

@app.post("/query")
def process_query(request: QueryRequest):
    try:
        # For NBA table visualization request
        if "Final_NBA_Output_python_20250519" in request.natural_language and "frequency" in request.natural_language.lower():
            # Generate SQL for frequency analysis
            if "marketing" in request.natural_language.lower() or "action" in request.natural_language.lower():
                sql = '''
                SELECT "Marketing_Action_Adj", COUNT(*) as frequency 
                FROM "Final_NBA_Output_python_20250519" 
                GROUP BY "Marketing_Action_Adj" 
                ORDER BY frequency DESC 
                LIMIT 10
                '''
            elif "recommend" in request.natural_language.lower() and "message" in request.natural_language.lower():
                sql = '''
                SELECT "Recommended_Msg_Overall", COUNT(*) as frequency 
                FROM "Final_NBA_Output_python_20250519" 
                WHERE "Recommended_Msg_Overall" != '{}' 
                GROUP BY "Recommended_Msg_Overall" 
                ORDER BY frequency DESC 
                LIMIT 10
                '''
            else:
                sql = '''
                SELECT "Marketing_Action_Adj", COUNT(*) as frequency 
                FROM "Final_NBA_Output_python_20250519" 
                GROUP BY "Marketing_Action_Adj" 
                ORDER BY frequency DESC 
                LIMIT 10
                '''
        else:
            raise HTTPException(status_code=400, detail="Query not supported")
        
        # Execute query
        result = adapter.run(sql)
        
        if result.error:
            raise HTTPException(status_code=500, detail=f"Query failed: {result.error}")
        
        # Create visualization
        labels = [row[0] for row in result.rows]
        values = [row[1] for row in result.rows]
        
        plotly_spec = {
            "data": [{
                "x": labels,
                "y": values,
                "type": "bar",
                "name": "Frequency",
                "marker": {"color": "#1f77b4"}
            }],
            "layout": {
                "title": "Frequency Analysis - NBA Data",
                "xaxis": {"title": "Category", "tickangle": -45},
                "yaxis": {"title": "Count"},
                "margin": {"l": 100, "r": 50, "t": 100, "b": 150}
            }
        }
        
        return {
            "sql": sql,
            "rows": result.rows,
            "row_count": len(result.rows),
            "execution_time": result.execution_time,
            "plotly_spec": plotly_spec,
            "success": True,
            "job_id": request.job_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-nba")
def test_nba():
    """Quick test of NBA table"""
    result = adapter.run('SELECT COUNT(*) FROM "Final_NBA_Output_python_20250519"')
    if result.error:
        return {"error": result.error}
    
    return {
        "table": "Final_NBA_Output_python_20250519",
        "total_records": result.rows[0][0],
        "status": "accessible"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting NBA Query Agent...")
    print(f"✅ Database: {type(adapter).__name__}")
    print("🔗 Server will be at: http://localhost:8002")
    uvicorn.run(app, host="127.0.0.1", port=8002)
