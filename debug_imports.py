#!/usr/bin/env python3
"""
Test script to debug the main.py import issues
"""
import sys
import os

# Add the project root and backend to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

print("🔧 Testing imports...")

try:
    print("Importing FastAPI...")
    from fastapi import FastAPI
    print("✅ FastAPI imported")
except Exception as e:
    print(f"❌ FastAPI import failed: {e}")
    sys.exit(1)

try:
    print("Importing db.engine...")
    from db.engine import get_adapter
    print("✅ db.engine imported")
except Exception as e:
    print(f"❌ db.engine import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("Importing db.schema...")
    from db.schema import get_schema_cache
    print("✅ db.schema imported")
except Exception as e:
    print(f"❌ db.schema import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("Testing get_adapter()...")
    adapter = get_adapter()
    print("✅ get_adapter() works")
except Exception as e:
    print(f"❌ get_adapter() failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("Testing get_schema_cache()...")
    schema_cache = get_schema_cache()
    print("✅ get_schema_cache() works")
except Exception as e:
    print(f"❌ get_schema_cache() failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("✅ All imports successful!")
