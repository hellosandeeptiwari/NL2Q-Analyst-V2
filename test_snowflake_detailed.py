#!/usr/bin/env python3
"""
Simple Snowflake Table Checker
Tests different ways to query tables in Snowflake
"""

import os
from dotenv import load_dotenv
import snowflake.connector

# Load environment variables
load_dotenv()

def test_snowflake_queries():
    print("🔍 Testing Snowflake table queries...")
    
    try:
        # Connect to Snowflake
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA"),
            role=os.getenv("SNOWFLAKE_ROLE")
        )
        
        cur = conn.cursor()
        
        # Test 1: Check current context
        print("\n1. Checking current context...")
        cur.execute("SELECT current_database(), current_schema(), current_warehouse(), current_role()")
        result = cur.fetchone()
        print(f"   Database: {result[0]}")
        print(f"   Schema: {result[1]}")
        print(f"   Warehouse: {result[2]}")
        print(f"   Role: {result[3]}")
        
        # Test 2: SHOW TABLES
        print("\n2. Testing SHOW TABLES...")
        try:
            cur.execute("SHOW TABLES")
            tables = cur.fetchall()
            print(f"   Found {len(tables)} tables with SHOW TABLES:")
            for i, table in enumerate(tables[:5]):
                print(f"     - {table[1]}")  # table[1] is the table name
            if len(tables) > 5:
                print(f"     ... and {len(tables) - 5} more")
        except Exception as e:
            print(f"   ❌ SHOW TABLES failed: {e}")
        
        # Test 3: INFORMATION_SCHEMA.TABLES
        print("\n3. Testing INFORMATION_SCHEMA.TABLES...")
        try:
            cur.execute("""
                SELECT TABLE_NAME, TABLE_SCHEMA
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
                LIMIT 10
            """)
            tables = cur.fetchall()
            print(f"   Found {len(tables)} tables with INFORMATION_SCHEMA:")
            for table in tables:
                print(f"     - {table[0]} (schema: {table[1]})")
        except Exception as e:
            print(f"   ❌ INFORMATION_SCHEMA.TABLES failed: {e}")
        
        # Test 4: Try broader INFORMATION_SCHEMA query
        print("\n4. Testing broader INFORMATION_SCHEMA query...")
        try:
            cur.execute("""
                SELECT TABLE_NAME, TABLE_SCHEMA
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA IS NOT NULL
                LIMIT 10
            """)
            tables = cur.fetchall()
            print(f"   Found {len(tables)} tables across all schemas:")
            for table in tables:
                print(f"     - {table[0]} (schema: {table[1]})")
        except Exception as e:
            print(f"   ❌ Broader INFORMATION_SCHEMA failed: {e}")
        
        # Test 5: Check available schemas
        print("\n5. Checking available schemas...")
        try:
            cur.execute("SHOW SCHEMAS")
            schemas = cur.fetchall()
            print(f"   Found {len(schemas)} schemas:")
            for schema in schemas[:10]:
                print(f"     - {schema[1]}")  # schema[1] is the schema name
        except Exception as e:
            print(f"   ❌ SHOW SCHEMAS failed: {e}")
        
        cur.close()
        conn.close()
        print("\n✅ Snowflake test completed!")
        
    except Exception as e:
        print(f"\n❌ Connection error: {e}")

if __name__ == "__main__":
    test_snowflake_queries()
