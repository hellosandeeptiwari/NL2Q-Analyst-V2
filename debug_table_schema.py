#!/usr/bin/env python3

"""
Debug script to check actual column names in the Final_NBA_Output_python table
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.config.snowflake_config import SnowflakeConnector

def debug_table_schema():
    """Check the actual column names in the table"""
    print("🔍 Debugging Final_NBA_Output_python table schema...")
    
    try:
        # Initialize Snowflake connector
        db_connector = SnowflakeConnector()
        
        # Table name
        table_name = "Final_NBA_Output_python_06042025"
        full_table_name = f'"COMMERCIAL_AI"."ENHANCED_NBA"."{table_name}"'
        
        print(f"📋 Checking table: {full_table_name}")
        
        # Method 1: DESCRIBE TABLE
        print("\n1️⃣ Using DESCRIBE TABLE:")
        try:
            describe_sql = f'DESCRIBE TABLE {full_table_name}'
            result = db_connector.run(describe_sql, dry_run=False)
            
            if result and result.rows:
                print(f"✅ Found {len(result.rows)} columns:")
                for i, row in enumerate(result.rows):
                    column_name = row[0]
                    data_type = row[1] if len(row) > 1 else "Unknown"
                    print(f"   {i+1}. {column_name} ({data_type})")
            else:
                print("❌ No columns found with DESCRIBE TABLE")
        except Exception as e:
            print(f"❌ DESCRIBE TABLE failed: {e}")
        
        # Method 2: Sample query to see actual data
        print("\n2️⃣ Using sample query:")
        try:
            sample_sql = f'SELECT * FROM {full_table_name} LIMIT 1'
            result = db_connector.run(sample_sql, dry_run=False)
            
            if result and result.column_names:
                print(f"✅ Found columns from sample query:")
                for i, col in enumerate(result.column_names):
                    print(f"   {i+1}. {col}")
            else:
                print("❌ No columns found with sample query")
                
            if result and result.rows:
                print(f"\n📊 Sample data (first row):")
                row = result.rows[0]
                for i, col in enumerate(result.column_names):
                    value = row[i] if i < len(row) else "N/A"
                    if isinstance(value, str) and len(value) > 50:
                        value = value[:50] + "..."
                    print(f"   {col}: {value}")
        except Exception as e:
            print(f"❌ Sample query failed: {e}")
            
        # Method 3: Check if recommended_msg column exists
        print("\n3️⃣ Checking for recommended_msg column:")
        try:
            # Try different case variations
            column_variations = [
                'recommended_msg',
                'RECOMMENDED_MSG', 
                'Recommended_Msg',
                'recommended_message',
                'RECOMMENDED_MESSAGE'
            ]
            
            for col_name in column_variations:
                try:
                    test_sql = f'SELECT "{col_name}" FROM {full_table_name} LIMIT 1'
                    result = db_connector.run(test_sql, dry_run=False)
                    print(f"   ✅ Column '{col_name}' exists!")
                    break
                except Exception:
                    print(f"   ❌ Column '{col_name}' not found")
        except Exception as e:
            print(f"❌ Column check failed: {e}")
            
    except Exception as e:
        print(f"❌ Database connection failed: {e}")

if __name__ == "__main__":
    debug_table_schema()
