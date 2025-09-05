import snowflake.connector
import os
from dotenv import load_dotenv
import sys
sys.path.append('backend')
from backend.db.engine import get_adapter

load_dotenv()

print("=== Testing Direct Connection (like our test) ===")
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
    print('✅ Direct connection successful!')
    
    cur = conn.cursor()
    cur.execute('SELECT * FROM "Final_NBA_Output_python_06042025" LIMIT 1')
    result = cur.fetchone()
    print('✅ Direct table SELECT works!')
    cur.close()
    conn.close()
    
except Exception as e:
    print(f'❌ Direct connection failed: {e}')

print("\n=== Testing Application Adapter (like our backend) ===")
try:
    adapter = get_adapter("snowflake")
    print('✅ Adapter created successfully!')
    
    # Test the exact same query that fails in our app
    sql = 'SELECT * FROM "Final_NBA_Output_python_06042025" LIMIT 1'
    result = adapter.run(sql, dry_run=False)
    
    if result.error:
        print(f'❌ Adapter query failed: {result.error}')
    else:
        print(f'✅ Adapter query works! Got {len(result.rows)} rows')
        print(f'Sample result: {result.rows[0] if result.rows else "No rows"}')
        
except Exception as e:
    print(f'❌ Adapter test failed: {e}')
    import traceback
    traceback.print_exc()

print("\n=== Testing Multiple Queries (connection reuse) ===")
try:
    adapter = get_adapter("snowflake")
    
    # Test multiple queries to see if connection gets stale
    for i in range(3):
        sql = f'SELECT {i+1} as test_num, * FROM "Final_NBA_Output_python_06042025" LIMIT 1'
        result = adapter.run(sql, dry_run=False)
        
        if result.error:
            print(f'❌ Query {i+1} failed: {result.error}')
            break
        else:
            print(f'✅ Query {i+1} works! Got {len(result.rows)} rows')
            
except Exception as e:
    print(f'❌ Multiple query test failed: {e}')
