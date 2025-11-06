"""
Check SQL comments for all indexed tables
"""
import asyncio
import os
from dotenv import load_dotenv
from backend.db.engine import AzureSQLAdapter

# Load environment variables
load_dotenv()

async def main():
    # Initialize database adapter with proper config
    print("✅ Connecting to Azure SQL Server directly...")
    config = {
        'host': os.getenv('AZURE_SQL_HOST'),
        'port': int(os.getenv('AZURE_SQL_PORT', '1433')),
        'user': os.getenv('AZURE_SQL_USER'),
        'password': os.getenv('AZURE_SQL_PASSWORD'),
        'dbname': os.getenv('AZURE_SQL_DATABASE')
    }
    db_adapter = AzureSQLAdapter(config)
    db_adapter.connect()
    
    # Verify connection
    if not db_adapter.conn:
        print("❌ Failed to connect to database")
        return
    
    # List of indexed tables
    indexed_tables = [
        "Reporting_BI_TerritoryPerformanceSummary",
        "Reporting_BI_Nrx_SampleSummary",
        "Reporting_BI_CallActivity",
        "Reporting_BI_NGD"
    ]
    
    print("="*80)
    print("CHECKING SQL COMMENTS FOR ALL INDEXED TABLES")
    print("="*80)
    
    for table_name in indexed_tables:
        print(f"\n{'='*80}")
        print(f"TABLE: {table_name}")
        print(f"{'='*80}")
        
        try:
            # Get table schema with comments
            columns = await db_adapter.get_table_schema(table_name)
            
            total_columns = len(columns)
            columns_with_comments = [col for col in columns if col.get('comment')]
            
            print(f"  Total columns: {total_columns}")
            print(f"  Columns with SQL comments: {len(columns_with_comments)}")
            print(f"  Coverage: {len(columns_with_comments)/total_columns*100:.1f}%")
            
            if columns_with_comments:
                print(f"\n  Sample columns with comments:")
                for i, col in enumerate(columns_with_comments[:5]):
                    comment = col.get('comment', '')
                    if len(comment) > 60:
                        comment = comment[:57] + "..."
                    print(f"    {i+1}. {col['name']}: {comment}")
            else:
                print(f"\n  ⚠️ NO SQL COMMENTS FOUND IN DATABASE")
            
            if len(columns_with_comments) < total_columns:
                columns_without = [col['name'] for col in columns if not col.get('comment')]
                print(f"\n  Columns WITHOUT comments ({len(columns_without)}):")
                print(f"    {', '.join(columns_without[:10])}")
                if len(columns_without) > 10:
                    print(f"    ... and {len(columns_without) - 10} more")
                
        except Exception as e:
            print(f"  ❌ Error getting table schema: {e}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Get summary
    for table_name in indexed_tables:
        try:
            columns = await db_adapter.get_table_schema(table_name)
            total = len(columns)
            with_comments = len([col for col in columns if col.get('comment')])
            status = "✅" if with_comments == total else "⚠️" if with_comments > 0 else "❌"
            print(f"  {status} {table_name}: {with_comments}/{total} columns ({with_comments/total*100:.0f}%)")
        except:
            print(f"  ❌ {table_name}: Error")

if __name__ == "__main__":
    asyncio.run(main())
