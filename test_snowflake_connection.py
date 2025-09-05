import snowflake.connector
import os
from dotenv import load_dotenv

load_dotenv()

try:
    conn = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA'),
        role=os.getenv('SNOWFLAKE_ROLE')
    )
    print('✅ Snowflake connection successful!')
    
    cur = conn.cursor()
    cur.execute('SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_DATABASE(), CURRENT_SCHEMA()')
    result = cur.fetchone()
    print(f'User: {result[0]}, Role: {result[1]}, Database: {result[2]}, Schema: {result[3]}')
    
    # Test a simple select
    cur.execute('SELECT 1 as test')
    print('✅ Simple SELECT works!')
    
    # Test selecting from the actual table with proper quoting
    table_name = '"Final_NBA_Output_python_06042025"'
    sql = f'SELECT * FROM {table_name} LIMIT 1'
    print(f'Testing SQL: {sql}')
    
    cur.execute(sql)
    result = cur.fetchone()
    print('✅ Table SELECT works! Got result:', result)
    
    conn.close()
    
except Exception as e:
    print(f'❌ Connection failed: {e}')
    import traceback
    traceback.print_exc()
