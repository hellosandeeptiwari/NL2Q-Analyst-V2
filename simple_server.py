#!/usr/bin/env python3
"""
Minimal FastAPI app to test NBA table query functionality
"""
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="NL2Q Agent", version="1.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import components with error handling
try:
    from db.engine import get_adapter
    adapter = get_adapter()
    print(f"✅ Database adapter initialized: {type(adapter).__name__}")
except Exception as e:
    print(f"❌ Database adapter failed: {e}")
    adapter = None

try:
    from nl2sql.generator import generate_sql
    print("✅ SQL generator imported")
except Exception as e:
    print(f"❌ SQL generator failed: {e}")
    generate_sql = None

try:
    from nl2sql.guardrails import GuardrailConfig
    print("✅ Guardrails imported")
except Exception as e:
    print(f"❌ Guardrails failed: {e}")
    GuardrailConfig = None

# Pydantic models
class QueryRequest(BaseModel):
    natural_language: str
    job_id: str = "default"
    db_type: str = "snowflake"

@app.get("/")
def root():
    return {"message": "NL2Q Agent is running", "adapter": type(adapter).__name__ if adapter else "None"}

@app.get("/health")
def health():
    if adapter:
        try:
            return adapter.health()
        except Exception as e:
            return {"error": str(e), "connected": False}
    return {"error": "No adapter available", "connected": False}

@app.post("/query")
def query_endpoint(request: QueryRequest):
    if not adapter:
        raise HTTPException(status_code=500, detail="Database adapter not available")
    
    if not generate_sql:
        raise HTTPException(status_code=500, detail="SQL generator not available")
    
    try:
        # Create guardrail config
        guardrail_cfg = GuardrailConfig(
            enable_write=False,
            allowed_schemas=["ENHANCED_NBA"],
            default_limit=100
        ) if GuardrailConfig else None
        
        # Generate SQL
        generated = generate_sql(request.natural_language, {}, guardrail_cfg) if guardrail_cfg else None
        
        if not generated:
            # Simple fallback for NBA table query
            if "Final_NBA_Output_python_20250519" in request.natural_language:
                sql = 'SELECT "Marketing_Action_Adj", COUNT(*) as frequency FROM "Final_NBA_Output_python_20250519" GROUP BY "Marketing_Action_Adj" ORDER BY frequency DESC LIMIT 10'
            else:
                raise HTTPException(status_code=400, detail="Could not generate SQL")
        else:
            sql = generated.sql
        
        # Execute query
        result = adapter.run(sql)
        
        if result.error:
            raise HTTPException(status_code=500, detail=f"Query execution failed: {result.error}")
        
        # Create simple visualization spec for frequency data
        plotly_spec = {
            "data": [{
                "x": [row[0] for row in result.rows[:10]],
                "y": [row[1] for row in result.rows[:10]],
                "type": "bar",
                "name": "Frequency"
            }],
            "layout": {
                "title": "Marketing Action Frequency",
                "xaxis": {"title": "Marketing Action"},
                "yaxis": {"title": "Count"}
            }
        }
        
        return {
            "sql": sql,
            "rows": result.rows[:100],
            "row_count": len(result.rows),
            "execution_time": result.execution_time,
            "plotly_spec": plotly_spec,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-nba")
def test_nba_query():
    """Test endpoint for NBA table query"""
    if not adapter:
        return {"error": "No database adapter"}
    
    try:
        # Test basic connection
        health = adapter.health()
        if not health.get("connected"):
            return {"error": "Database not connected", "health": health}
        
        # Test table access
        sql = 'SELECT COUNT(*) FROM "Final_NBA_Output_python_20250519" LIMIT 1'
        result = adapter.run(sql)
        
        if result.error:
            return {"error": f"Table access failed: {result.error}"}
        
        count = result.rows[0][0] if result.rows else 0
        
        # Test sample data
        sql_sample = 'SELECT "Marketing_Action_Adj", "Recommended_Msg_Overall" FROM "Final_NBA_Output_python_20250519" LIMIT 5'
        result_sample = adapter.run(sql_sample)
        
        return {
            "table_accessible": True,
            "total_records": count,
            "sample_data": result_sample.rows if not result_sample.error else [],
            "health": health
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting NL2Q Agent on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
