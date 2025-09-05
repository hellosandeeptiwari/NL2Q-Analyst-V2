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
    
    cur = conn.cursor()
    cur.execute('SELECT * FROM "Final_NBA_Output_python_06042025" LIMIT 1')
    
    # Get column descriptions
    columns = [desc[0] for desc in cur.description]
    print(f"Column names: {columns}")
    
    result = cur.fetchone()
    print(f"Data: {result}")
    
    # Show as dictionary
    row_dict = {columns[i]: result[i] for i in range(len(columns))}
    print(f"As dict: {row_dict}")
    
    cur.close()
    conn.close()
    
except Exception as e:
    print(f'❌ Error: {e}')
