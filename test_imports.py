#!/usr/bin/env python3
"""
Minimal test to check if the backend modules can be imported correctly
"""
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

print("🔧 Loading environment variables...")
from dotenv import load_dotenv
load_dotenv()

print(f"DB_ENGINE: {os.getenv('DB_ENGINE')}")
print(f"SNOWFLAKE_USER: {os.getenv('SNOWFLAKE_USER')}")
print(f"SNOWFLAKE_DATABASE: {os.getenv('SNOWFLAKE_DATABASE')}")

print("\n🔧 Testing backend imports...")

try:
    from db.engine import get_adapter
    print("✅ db.engine imported successfully")
except Exception as e:
    print(f"❌ Failed to import db.engine: {e}")
    sys.exit(1)

try:
    from db.schema import get_schema_cache
    print("✅ db.schema imported successfully")
except Exception as e:
    print(f"❌ Failed to import db.schema: {e}")
    sys.exit(1)

try:
    from nl2sql.bias_detection import BiasDetector
    print("✅ nl2sql.bias_detection imported successfully")
except Exception as e:
    print(f"❌ Failed to import nl2sql.bias_detection: {e}")
    sys.exit(1)

try:
    from nl2sql.guardrails import GuardrailConfig
    print("✅ nl2sql.guardrails imported successfully")
except Exception as e:
    print(f"❌ Failed to import nl2sql.guardrails: {e}")
    sys.exit(1)

try:
    from agent.pipeline import NLQueryNode
    print("✅ agent.pipeline imported successfully")
except Exception as e:
    print(f"❌ Failed to import agent.pipeline: {e}")
    sys.exit(1)

try:
    from audit.audit_log import log_audit
    print("✅ audit.audit_log imported successfully")
except Exception as e:
    print(f"❌ Failed to import audit.audit_log: {e}")
    sys.exit(1)

try:
    from exports.csv_export import to_csv
    print("✅ exports.csv_export imported successfully")
except Exception as e:
    print(f"❌ Failed to import exports.csv_export: {e}")
    sys.exit(1)

try:
    from storage.data_storage import DataStorage
    print("✅ storage.data_storage imported successfully")
except Exception as e:
    print(f"❌ Failed to import storage.data_storage: {e}")
    sys.exit(1)

try:
    from auth.auth import verify_token
    print("✅ auth.auth imported successfully")
except Exception as e:
    print(f"❌ Failed to import auth.auth: {e}")
    sys.exit(1)

try:
    from history.query_history import save_query_history, get_recent_queries
    print("✅ history.query_history imported successfully")
except Exception as e:
    print(f"❌ Failed to import history.query_history: {e}")
    sys.exit(1)

try:
    from analytics.usage import log_usage
    print("✅ analytics.usage imported successfully")
except Exception as e:
    print(f"❌ Failed to import analytics.usage: {e}")
    sys.exit(1)

try:
    from errors.error_reporting import report_error, get_error_reports
    print("✅ errors.error_reporting imported successfully")
except Exception as e:
    print(f"❌ Failed to import errors.error_reporting: {e}")
    sys.exit(1)

print("\n🎉 All backend modules imported successfully!")
print("🔧 Testing database adapter initialization...")

try:
    adapter = get_adapter()
    print("✅ Database adapter initialized successfully")
    print(f"Adapter type: {type(adapter).__name__}")
except Exception as e:
    print(f"❌ Failed to initialize database adapter: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ Backend is ready for testing!")
