#!/usr/bin/env python3
"""
Direct Snowflake Connection Test
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

def test_direct_snowflake():
    """Test direct Snowflake connection"""
    
    print("🔌 Testing Direct Snowflake Connection...")
    
    # Check DB_ENGINE
    db_engine = os.getenv("DB_ENGINE", "not_set")
    print(f"🔧 DB_ENGINE: {db_engine}")
    
    if db_engine != "snowflake":
        print(f"❌ DB_ENGINE not set to snowflake")
        return False
    
    try:
        # Import and test direct connection
        import snowflake.connector
        
        config = {
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA")
        }
        
        print(f"🔧 Connecting with config:")
        for key, value in config.items():
            if 'password' in key.lower():
                print(f"   {key}: ***hidden***")
            else:
                print(f"   {key}: {value}")
        
        # Direct connection
        conn = snowflake.connector.connect(**config)
        
        # Test query
        cur = conn.cursor()
        cur.execute("SELECT 1 as test")
        result = cur.fetchone()
        cur.close()
        
        print(f"✅ Direct Snowflake connection successful!")
        print(f"📊 Test query result: {result}")
        
        # Test NBA table access
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM NBA_PHASE2_SEP2024_SIMILARITY_OUTPUT_FINAL_PYTHON LIMIT 1")
        count_result = cur.fetchone()
        cur.close()
        
        print(f"✅ NBA table accessible!")
        print(f"📊 NBA table count: {count_result}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Direct Snowflake connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adapter_snowflake():
    """Test through adapter"""
    
    print(f"\n🔌 Testing Snowflake Through Adapter...")
    
    try:
        from backend.db.engine import get_adapter
        
        # Force snowflake
        adapter = get_adapter("snowflake")
        print(f"✅ Adapter created: {type(adapter)}")
        
        # Test health
        health = adapter.health()
        print(f"📊 Health check: {health}")
        
        if health.get('connected'):
            print(f"✅ Adapter connection successful!")
            
            # Test query
            result = adapter.run("SELECT 1 as test", dry_run=False)
            if result.error:
                print(f"❌ Test query failed: {result.error}")
            else:
                print(f"✅ Test query successful: {result.rows}")
                
            return True
        else:
            print(f"❌ Adapter connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 SNOWFLAKE CONNECTION DEBUGGING")
    print("="*50)
    
    # Test 1: Direct connection
    direct_success = test_direct_snowflake()
    
    # Test 2: Through adapter
    adapter_success = test_adapter_snowflake()
    
    print(f"\n📋 RESULTS:")
    print(f"   Direct connection: {'✅ SUCCESS' if direct_success else '❌ FAILED'}")
    print(f"   Adapter connection: {'✅ SUCCESS' if adapter_success else '❌ FAILED'}")
    
    if direct_success and adapter_success:
        print(f"\n🎉 Snowflake connection working!")
        print(f"🔄 Ready to run end-to-end NL2Q test")
    else:
        print(f"\n🔧 Connection issues need to be resolved")
