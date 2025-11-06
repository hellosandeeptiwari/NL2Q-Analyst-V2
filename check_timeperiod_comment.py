"""
Check TimePeriod column comment
"""
from backend.db.engine import get_adapter

adapter = get_adapter()

# Check SQL comment for TimePeriod column
result = adapter.run("""
SELECT 
    c.name AS ColumnName, 
    ep.value AS Description 
FROM sys.columns c 
LEFT JOIN sys.extended_properties ep 
    ON ep.major_id = c.object_id 
    AND ep.minor_id = c.column_id 
    AND ep.name = 'MS_Description' 
WHERE c.object_id = OBJECT_ID('Reporting_BI_Nrx_SampleSummary') 
    AND c.name = 'TimePeriod'
""", dry_run=False)

print('TimePeriod column comment:')
if result.rows:
    for row in result.rows:
        print(f'  Column: {row[0]}')
        print(f'  Description: {row[1]}')
else:
    print('  No comment found')

# Also check what values exist
result2 = adapter.run("""
SELECT DISTINCT TOP 10 TimePeriod 
FROM Reporting_BI_Nrx_SampleSummary 
ORDER BY TimePeriod DESC
""", dry_run=False)

print('\nActual TimePeriod values in database:')
for row in result2.rows:
    print(f'  - {row[0]}')
