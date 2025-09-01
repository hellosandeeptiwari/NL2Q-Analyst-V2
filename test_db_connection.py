import sys
from pathlib import Path
backend_path = Path('.') / 'backend'
sys.path.insert(0, str(backend_path))

from dotenv import load_dotenv
load_dotenv()

print('Testing database connection...')
from db.engine import get_adapter
adapter = get_adapter()
print(f'Adapter: {type(adapter).__name__}')

health = adapter.health()
print(f'Health: {health}')

if health.get('connected'):
    print('Testing NBA table...')
    result = adapter.run('SELECT COUNT(*) FROM "Final_NBA_Output_python_20250519"')
    if result.error:
        print(f'Error: {result.error}')
    else:
        print(f'Table has {result.rows[0][0]} records')
        
    # Test sample data
    print('Getting sample data...')
    result = adapter.run('SELECT "Marketing_Action_Adj", COUNT(*) as freq FROM "Final_NBA_Output_python_20250519" GROUP BY "Marketing_Action_Adj" ORDER BY freq DESC LIMIT 5')
    if result.error:
        print(f'Sample query error: {result.error}')
    else:
        print('Sample frequency data:')
        for row in result.rows:
            print(f'  {row[0]}: {row[1]}')
else:
    print('Database not connected')
