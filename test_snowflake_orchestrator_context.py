#!/usr/bin/env python3

import os
import asyncio
from backend.db.engine import get_adapter

async def test_snowflake_connection_in_orchestrator_context():
    """Test Snowflake connection in orchestrator-like context"""
    
    print("🔍 Testing Snowflake connection in orchestrator context...")
    
    # Test 1: Direct adapter connection
    try:
        print("\n1. Testing direct get_adapter connection...")
        adapter = get_adapter("snowflake")
        result = adapter.run("SELECT * FROM \"Final_NBA_Output_python_06042025\" LIMIT 1", dry_run=False)
        if result.error:
            print(f"❌ Direct adapter failed: {result.error}")
        else:
            print(f"✅ Direct adapter success: {len(result.rows)} rows")
    except Exception as e:
        print(f"❌ Direct adapter exception: {e}")
    
    # Test 2: Test environment variables
    print("\n2. Testing environment variables...")
    required_vars = ['SNOWFLAKE_USER', 'SNOWFLAKE_PASSWORD', 'SNOWFLAKE_ACCOUNT', 
                     'SNOWFLAKE_WAREHOUSE', 'SNOWFLAKE_DATABASE', 'SNOWFLAKE_SCHEMA', 'SNOWFLAKE_ROLE']
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {'*' * len(value) if 'PASSWORD' in var else value}")
        else:
            print(f"❌ {var}: Not set")
    
    # Test 3: Test with fresh adapter creation (like orchestrator does)
    try:
        print("\n3. Testing fresh adapter creation...")
        from backend.db.engine import SnowflakeAdapter
        
        config = {
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA"),
            "role": os.getenv("SNOWFLAKE_ROLE")
        }
        
        adapter = SnowflakeAdapter(config)
        adapter.connect()
        
        result = adapter.run("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_ROLE()", dry_run=False)
        if result.error:
            print(f"❌ Fresh adapter failed: {result.error}")
        else:
            print(f"✅ Fresh adapter success: {result.rows}")
            
    except Exception as e:
        print(f"❌ Fresh adapter exception: {e}")
    
    print("\n✅ Snowflake connection test completed")

if __name__ == "__main__":
    asyncio.run(test_snowflake_connection_in_orchestrator_context())
