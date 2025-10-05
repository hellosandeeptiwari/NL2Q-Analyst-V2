import asyncio
import sys
sys.path.append('backend')

from backend.db.connector import DatabaseConnector

async def check_table_info():
    # Initialize connector
    db = DatabaseConnector()
    
    # Check if Reporting_BI_PrescriberProfile is a view or table
    query = """
    SELECT 
        TABLE_SCHEMA,
        TABLE_NAME, 
        TABLE_TYPE
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_NAME IN ('Reporting_BI_PrescriberProfile', 'Reporting_BI_PrescriberOverview', 'Reporting_BI_NGD')
    """
    
    result = db.run(query)
    
    print("Table types:")
    for row in result.rows:
        print(f"  {row[1]}: {row[2]}")
    
    # Now check TRX column specifically - actual data samples
    query2 = """
    SELECT TOP 10
        TRX,
        ISNUMERIC(TRX) as IsNumeric,
        TRY_CAST(TRX AS INT) as AsInt,
        TYPEOF(TRX) as TypeOf
    FROM Reporting_BI_PrescriberProfile
    WHERE TRX IS NOT NULL
    """
    
    try:
        result2 = db.run(query2)
        
        print("\nTRX actual data samples:")
        for row in result2.rows:
            print(f"  Value: {row[0]}, IsNumeric: {row[1]}, AsInt: {row[2]}")
    except Exception as e:
        print(f"\nError querying TRX data: {e}")
        
        # Try simpler query
        query3 = """
        SELECT TOP 5 TRX
        FROM Reporting_BI_PrescriberProfile
        WHERE TRX IS NOT NULL
        """
        result3 = db.run(query3)
        print("\nTRX simple samples:")
        for row in result3.rows:
            print(f"  Value: '{row[0]}' (type: {type(row[0])})")

asyncio.run(check_table_info())
