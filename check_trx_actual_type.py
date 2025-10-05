import pyodbc

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=odsdevserver.database.windows.net;'
    'DATABASE=DWHDevIBSA_C_18082104202;'
    'UID=odsdevserver_sqladmin;'
    'PWD=odsDevServer_18Aug21#'
)

cursor = conn.cursor()
cursor.execute("""
    SELECT COLUMN_NAME, DATA_TYPE 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME='Reporting_BI_PrescriberProfile' 
    AND COLUMN_NAME IN ('TRX', 'NRX', 'TotalCalls')
    ORDER BY COLUMN_NAME
""")

print("Actual database column types:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]}")

conn.close()
