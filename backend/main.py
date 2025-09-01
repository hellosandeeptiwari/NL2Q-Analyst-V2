from fastapi import FastAPI, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from db.engine import get_adapter
from db.schema import get_schema_cache
from nl2sql.bias_detection import BiasDetector
from nl2sql.guardrails import GuardrailConfig
from agent.pipeline import NLQueryNode
from audit.audit_log import log_audit
from exports.csv_export import to_csv
from storage.data_storage import DataStorage
from auth.auth import verify_token
from history.query_history import save_query_history, get_recent_queries
from analytics.usage import log_usage
from errors.error_reporting import report_error, get_error_reports

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

adapter = get_adapter()
schema_cache = get_schema_cache()
storage = DataStorage(os.getenv("STORAGE_TYPE", "local"))
bias_detector = BiasDetector()

@app.get("/health")
def health():
    log_usage("/health")
    return adapter.health()

@app.get("/schema")
def schema():
    log_usage("/schema")
    return JSONResponse(schema_cache)

@app.post("/query")
async def query(request: Request, token: str = Depends(verify_token)):
    log_usage("/query")
    body = await request.json()
    nl = body.get("natural_language")
    job_id = body.get("job_id")
    db_type = body.get("db_type", os.getenv("DB_ENGINE", "sqlite"))
    try:
        guardrail_cfg = GuardrailConfig(
            enable_write=False,
            allowed_schemas=["public"],
            default_limit=100
        )
        generated = generate_sql(nl, schema_cache, guardrail_cfg)
        if db_type != os.getenv("DB_ENGINE", "sqlite"):
            adapter = get_adapter(db_type)
        else:
            adapter = get_adapter()
        result = adapter.run(generated.sql)
        location = storage.save_data(result.rows, job_id)
        log_audit(nl, generated.sql, result.execution_time, len(result.rows), result.error)
        save_query_history(nl, generated.sql, job_id)
        import pandas as pd
        df = pd.DataFrame(result.rows)
        plotly_spec = result.plotly_spec if hasattr(result, 'plotly_spec') else {}
        bias_report = bias_detector.detect_bias(result.rows, nl)
        return {
            "sql": generated.sql,
            "rows": result.rows[:100],  # Limit for response
            "location": location,
            "plotly_spec": plotly_spec,
            "suggestions": generated.suggestions,
            "bias_report": bias_report
        }
    except Exception as e:
        report_error(str(e), {"nl": nl, "job_id": job_id})
        raise

@app.get("/csv/{job_id}")
def get_csv(job_id: str, token: str = Depends(verify_token)):
    log_usage("/csv")
    location = storage.save_data([], job_id)  # Retrieve location
    if location.startswith("s3://"):
        # Redirect to S3
        return {"url": location}
    elif location.startswith("sharepoint://"):
        # Handle SharePoint
        return {"url": location}
    else:
        return StreamingResponse(open(location, "rb"), media_type="text/csv")

@app.post("/insights")
async def insights(request: Request, token: str = Depends(verify_token)):
    log_usage("/insights")
    body = await request.json()
    location = body.get("location")
    query = body.get("query")
    try:
        data = storage.load_data(location)
        insight = storage.generate_insights(data, query)
        return {"insight": insight}
    except Exception as e:
        report_error(str(e), {"location": location, "query": query})
        raise

@app.get("/history")
def history(token: str = Depends(verify_token)):
    log_usage("/history")
    return JSONResponse(get_recent_queries())

@app.get("/analytics")
def analytics(token: str = Depends(verify_token)):
    log_usage("/analytics")
    from backend.analytics.usage import get_usage_stats
    return JSONResponse(get_usage_stats())

@app.get("/errors")
def errors(token: str = Depends(verify_token)):
    log_usage("/errors")
    return JSONResponse(get_error_reports())

@app.get("/events/status")
async def sse_status(request: Request):
    async def event_stream():
        while True:
            if await request.is_disconnected():
                break
            yield f"data: {json.dumps(adapter.health())}\n\n"
            await asyncio.sleep(2)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/logs")
def logs(token: str = Depends(verify_token)):
    log_usage("/logs")
    with open("backend/audit/audit_log.jsonl", "r", encoding="utf-8") as f:
        return JSONResponse([json.loads(line) for line in f])

if __name__ == "__main__":
    print("🚀 Starting uvicorn server...")
    import uvicorn
    try:
        print("📡 Running uvicorn on port 8001...")
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
