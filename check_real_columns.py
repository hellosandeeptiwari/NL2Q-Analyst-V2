#!/usr/bin/env python3
"""
Direct database query to check actual column names
"""

import pyodbc
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("🔍 Checking actual Azure SQL Server columns")
    print("=" * 80)
    
    try:
        # Connection parameters
        server = os.getenv('AZURE_SQL_SERVER', 'odsdevserver.database.windows.net')
        database = os.getenv('AZURE_SQL_DATABASE', 'DWHDevIBSA_C_18082104202')
        username = os.getenv('AZURE_SQL_USERNAME', 'DWHDevIBSAJbsUsrC4202')
        password = os.getenv('AZURE_SQL_PASSWORD')
        
        if not password:
            print("❌ No password found in environment variables")
            return
        
        # Connection string
        conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
        
        print(f"🔗 Connecting to: {server}")
        print(f"📊 Database: {database}")
        
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            
            # Test tables
            tables = ['Reporting_BI_NGD', 'Reporting_BI_PrescriberOverview', 'Reporting_BI_PrescriberProfile']
            
            for table in tables:
                print(f"\n📊 TABLE: {table}")
                print("-" * 50)
                
                try:
                    # Get column information using INFORMATION_SCHEMA
                    query = """
                    SELECT COLUMN_NAME, DATA_TYPE 
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME = ?
                    ORDER BY ORDINAL_POSITION
                    """
                    cursor.execute(query, table)
                    columns = cursor.fetchall()
                    
                    if columns:
                        print(f"✅ Found {len(columns)} columns:")
                        for i, (col_name, data_type) in enumerate(columns):
                            print(f"  {i+1:2d}. {col_name} ({data_type})")
                    else:
                        print("❌ No columns found - table may not exist")
                        
                except Exception as e:
                    print(f"❌ Error querying {table}: {e}")
            
            # Check for specific problematic columns
            print(f"\n🔍 Checking for specific problematic columns:")
            print("-" * 50)
            
            problem_columns = ['Territory', 'PerformanceMetric', 'RepID', 'RepName', 'TerritoryName']
            
            for col in problem_columns:
                try:
                    query = """
                    SELECT TABLE_NAME, COLUMN_NAME 
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE COLUMN_NAME = ?
                    """
                    cursor.execute(query, col)
                    results = cursor.fetchall()
                    
                    if results:
                        print(f"✅ {col} found in:")
                        for table_name, column_name in results:
                            print(f"    - {table_name}")
                    else:
                        print(f"❌ {col} - NOT FOUND in any table")
                        
                except Exception as e:
                    print(f"❌ Error checking {col}: {e}")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()