import sqlite3
import os

db_path = 'schema_cache.db'
if not os.path.exists(db_path):
    print(f"Schema cache not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# First check what tables exist
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables in schema_cache.db:")
if not tables:
    print("  NO TABLES FOUND!")
else:
    for table in tables:
        print(f"  {table[0]}")
        
        # Get first few rows from each table
        try:
            cursor.execute(f"SELECT * FROM {table[0]} LIMIT 2")
            rows = cursor.fetchall()
            cursor.execute(f"PRAGMA table_info({table[0]})")
            columns = [col[1] for col in cursor.fetchall()]
            print(f"    Columns: {', '.join(columns)}")
            for row in rows:
                print(f"    Sample: {row}")
        except Exception as e:
            print(f"    Error: {e}")

conn.close()
