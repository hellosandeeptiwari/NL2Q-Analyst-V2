#!/usr/bin/env python3
"""
Test script to check if the problematic tables exist and their exact schema
"""

import os
import pyodbc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_table_queries():
    """Test different ways to reference the tables"""
    
    # Connection parameters
    config = {
        "host": os.getenv("AZURE_SQL_HOST"),
        "port": os.getenv("AZURE_SQL_PORT", 1433),
        "user": os.getenv("AZURE_SQL_USER"),
        "password": os.getenv("AZURE_SQL_PASSWORD"),
        "dbname": os.getenv("AZURE_SQL_DATABASE")
    }
    
    connection_string = f"""
    DRIVER={{ODBC Driver 17 for SQL Server}};
    SERVER={config['host']},{config['port']};
    DATABASE={config['dbname']};
    UID={config['user']};
    PWD={config['password']};
    Timeout=10;
    """
    
    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        
        # Test different table reference patterns
        test_queries = [
            "SELECT TOP 1 * FROM Reporting_BI_NGD",
            "SELECT TOP 1 * FROM [Reporting_BI_NGD]",
            "SELECT TOP 1 * FROM dbo.Reporting_BI_NGD",
            "SELECT TOP 1 * FROM [dbo].[Reporting_BI_NGD]",
        ]
        
        print("🧪 Testing different table reference patterns...\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"Test {i}: {query}")
            try:
                cursor.execute(query)
                result = cursor.fetchone()
                print(f"   ✅ SUCCESS: Query executed, got {len(result) if result else 0} columns")
                if result:
                    print(f"   📊 Sample columns: {result[:3]}...")
                break  # Found working pattern
            except pyodbc.Error as e:
                print(f"   ❌ FAILED: {str(e)[:100]}...")
            print()
        
        # Also test the other tables
        print("\n🧪 Testing other tables...")
        other_tables = [
            "Reporting_BI_PrescriberOverview",
            "Reporting_BI_PrescriberProfile"
        ]
        
        for table in other_tables:
            try:
                cursor.execute(f"SELECT TOP 1 * FROM {table}")
                result = cursor.fetchone()
                print(f"✅ {table}: {len(result) if result else 0} columns")
            except pyodbc.Error as e:
                print(f"❌ {table}: {str(e)[:50]}...")
        
        # Check schema information
        print("\n🔍 Checking table schema information...")
        schema_query = """
        SELECT 
            TABLE_SCHEMA,
            TABLE_NAME,
            TABLE_TYPE
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_NAME LIKE 'Reporting_BI_%'
        ORDER BY TABLE_NAME
        """
        
        cursor.execute(schema_query)
        results = cursor.fetchall()
        
        print(f"Found {len(results)} Reporting_BI tables:")
        for row in results:
            schema, name, table_type = row
            print(f"  📋 {schema}.{name} ({table_type})")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    test_table_queries()