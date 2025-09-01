from dotenv import load_dotenv
import snowflake.connector
import os

load_dotenv()

config = {
    'user': os.getenv('SNOWFLAKE_USER'),
    'password': os.getenv('SNOWFLAKE_PASSWORD'),
    'account': os.getenv('SNOWFLAKE_ACCOUNT'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'database': os.getenv('SNOWFLAKE_DATABASE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA'),
    'role': os.getenv('SNOWFLAKE_ROLE')
}

conn = snowflake.connector.connect(**config)
cur = conn.cursor()

print('Available tables in ENHANCED_NBA schema:')
cur.execute('SHOW TABLES')
tables = cur.fetchall()

print(f'Total tables found: {len(tables)}')

# Look for NBA-related tables
nba_tables = []
for table in tables:
    table_name = table[1]
    if 'NBA' in table_name.upper():
        nba_tables.append(table_name)

print(f'\nNBA-related tables ({len(nba_tables)}):')
for i, table in enumerate(nba_tables):
    print(f'  {i+1}. {table}')

# Check if the specific table exists
target_table = 'Final_NBA_Output_python_20250519'
if target_table in nba_tables:
    print(f'\n✅ Found target table: {target_table}')

    try:
        # Use quoted identifier for Snowflake case sensitivity
        cur.execute(f'DESCRIBE TABLE "{target_table}"')
        columns = cur.fetchall()
        print(f'\nTable structure ({len(columns)} columns):')
        for i, col in enumerate(columns[:15]):  # Show first 15 columns
            print(f'  {i+1}. {col[0]}: {col[1]}')

        # Get sample data
        cur.execute(f'SELECT * FROM "{target_table}" LIMIT 5')
        rows = cur.fetchall()
        print(f'\nSample data (5 rows):')
        for i, row in enumerate(rows):
            print(f'Row {i+1}: {row[:8]}...')  # Show first 8 columns

        # Get total count
        cur.execute(f'SELECT COUNT(*) FROM "{target_table}"')
        total_count = cur.fetchone()[0]
        print(f'\nTotal records: {total_count}')

    except Exception as e:
        print(f'Error accessing table: {e}')

else:
    print(f'\n❌ Target table "{target_table}" not found')
    print('\nAvailable alternatives:')
    for table in nba_tables[:5]:  # Show first 5 alternatives
        print(f'  - {table}')

cur.close()
conn.close()
