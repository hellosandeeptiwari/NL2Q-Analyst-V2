#!/usr/bin/env python3
"""
Debug script to identify what's causing the server to shut down
"""
import sys
import os
import traceback

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

print("🔧 Loading environment variables...")
from dotenv import load_dotenv
load_dotenv()

print("🔧 Testing step-by-step initialization...")

try:
    print("1. Importing FastAPI...")
    from fastapi import FastAPI
    print("✅ FastAPI imported")

    print("2. Creating FastAPI app...")
    app = FastAPI(title="NL2Q Agent", version="1.0.0")
    print("✅ App created")

    print("3. Importing database adapter...")
    from db.engine import get_adapter
    print("✅ Database engine imported")

    print("4. Initializing database adapter...")
    adapter = get_adapter()
    print(f"✅ Adapter initialized: {type(adapter).__name__}")

    print("5. Testing database connection...")
    health = adapter.health()
    print(f"✅ Database health: {health}")

    print("6. Importing other modules...")
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
    print("✅ All modules imported")

    print("7. Setting up CORS...")
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    print("✅ CORS configured")

    print("8. Initializing components...")
    schema_cache = get_schema_cache()
    bias_detector = BiasDetector()
    storage = DataStorage(os.getenv("STORAGE_TYPE", "local"))
    print("✅ Components initialized")

    print("9. Adding routes...")
    @app.get("/health")
    def health_endpoint():
        return adapter.health()

    @app.get("/test")
    def test_endpoint():
        return {"status": "ok", "adapter": type(adapter).__name__}

    print("✅ Routes added")

    print("\n🎉 All initialization steps completed successfully!")
    print("🔧 The issue might be in the route handlers or middleware.")

except Exception as e:
    print(f"\n❌ Error occurred at step: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)
