import asyncio
import os
from dotenv import load_dotenv
from backend.db.engine import AzureSQLAdapter

# Load environment variables
load_dotenv()

async def check_territory_table():
    """Check if Reporting_BI_TerritoryPerformanceSummary table has comments"""
    
    # Create Azure SQL adapter
    config = {
        'host': os.getenv('AZURE_SQL_HOST'),
        'dbname': os.getenv('AZURE_SQL_DATABASE'),
        'user': os.getenv('AZURE_SQL_USER'),
        'password': os.getenv('AZURE_SQL_PASSWORD'),
        'port': os.getenv('AZURE_SQL_PORT', '1433')
    }
    adapter = AzureSQLAdapter(config)
    
    # Connect to database
    adapter.connect()
    
    print("Searching for Reporting_BI_TerritoryPerformanceSummary table...\n")
    
    # Get all tables and find our target
    tables = await adapter.get_table_names()
    target_table = None
    for table in tables:
        if table.get('name') == 'Reporting_BI_TerritoryPerformanceSummary':
            target_table = table
            break
    
    if not target_table:
        print("❌ Table 'Reporting_BI_TerritoryPerformanceSummary' not found in database")
        return
    
    print("✅ Table found!")
    print(f"Table Name: {target_table.get('name')}")
    print(f"Table Comment: {target_table.get('comment') or '⚠️ No table comment found'}")
    print(f"\n{'='*80}")
    
    # Get column schema with comments
    columns = await adapter.get_table_schema('Reporting_BI_TerritoryPerformanceSummary')
    
    print(f"\nTotal Columns: {len(columns)}")
    print(f"\n{'='*80}")
    print("Column Details (showing all columns):\n")
    
    columns_with_comments = 0
    for i, col in enumerate(columns, 1):
        col_name = col.get('name')
        col_type = col.get('type', col.get('data_type', 'unknown'))
        col_comment = col.get('comment')
        
        if col_comment:
            columns_with_comments += 1
            print(f"{i}. {col_name} ({col_type})")
            print(f"   💬 Comment: {col_comment}")
        else:
            print(f"{i}. {col_name} ({col_type})")
            print(f"   ⚠️ No comment")
        print()
    
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  - Total columns: {len(columns)}")
    print(f"  - Columns with comments: {columns_with_comments}")
    print(f"  - Columns without comments: {len(columns) - columns_with_comments}")
    print(f"  - Table comment: {'Yes' if target_table.get('comment') else 'No'}")

if __name__ == "__main__":
    asyncio.run(check_territory_table())
