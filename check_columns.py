from backend.db.engine import get_adapter

def get_actual_columns():
    adapter = get_adapter()
    result = adapter.run('DESCRIBE TABLE "COMMERCIAL_AI"."ENHANCED_NBA"."Final_NBA_Output_python_06042025"')
    print('Actual columns in table:')
    for i, row in enumerate(result.rows[:15]):
        print(f'{i+1}. {row[0]} ({row[1]})')
    if len(result.rows) > 15:
        print(f'... and {len(result.rows) - 15} more columns')
    return [row[0] for row in result.rows]

if __name__ == "__main__":
    columns = get_actual_columns()
    print(f"\nTotal columns: {len(columns)}")
    print("\nLooking for columns with 'message' or 'input':")
    for col in columns:
        if 'message' in col.lower() or 'input' in col.lower():
            print(f"- {col}")
