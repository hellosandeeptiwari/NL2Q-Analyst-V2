import os
import sqlite3

print("Database Connection Test")
print("=======================")

# Check if nba.db exists
db_path = "nba.db"
print(f"Checking if database file {db_path} exists...")

if os.path.exists(db_path):
    print(f"✅ Database file found: {os.path.abspath(db_path)}")
else:
    print(f"❌ Database file not found at {os.path.abspath(db_path)}")
    print("Creating a sample database for testing...")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create a test table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_table (
            id INTEGER PRIMARY KEY,
            name TEXT,
            value INTEGER
        )
        ''')
        
        # Insert some test data
        cursor.execute("INSERT INTO test_table (name, value) VALUES (?, ?)", ("Test 1", 100))
        cursor.execute("INSERT INTO test_table (name, value) VALUES (?, ?)", ("Test 2", 200))
        
        conn.commit()
        print(f"✅ Created test database with sample data at {os.path.abspath(db_path)}")
    except Exception as e:
        print(f"❌ Failed to create test database: {e}")
        exit(1)

# Try to connect and run a query
try:
    print("\nTrying to connect to the database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    if tables:
        print(f"✅ Successfully connected! Found {len(tables)} tables:")
        for table in tables:
            print(f"  - {table[0]}")
            
            # Show first few rows of each table
            try:
                cursor.execute(f"SELECT * FROM {table[0]} LIMIT 3")
                rows = cursor.fetchall()
                if rows:
                    print(f"    Sample data: {rows[0]}")
            except Exception as e:
                print(f"    Could not get sample data: {e}")
    else:
        print("✅ Connected to database but no tables found.")
        
except Exception as e:
    print(f"❌ Failed to connect to database: {e}")

print("\nTest completed!")
