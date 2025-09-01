import sqlite3

# Create a sample SQLite database for testing
conn = sqlite3.connect('backend/db/nl2q.db')
cur = conn.cursor()

# Create sample tables
cur.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    age INTEGER,
    gender TEXT
)
''')

cur.execute('''
CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY,
    product TEXT,
    amount REAL,
    date TEXT,
    user_id INTEGER
)
''')

# Insert sample data
cur.execute("INSERT OR IGNORE INTO users (id, name, email, age, gender) VALUES (1, 'Alice', 'alice@example.com', 30, 'female')")
cur.execute("INSERT OR IGNORE INTO users (id, name, email, age, gender) VALUES (2, 'Bob', 'bob@example.com', 25, 'male')")
cur.execute("INSERT OR IGNORE INTO users (id, name, email, age, gender) VALUES (3, 'Charlie', 'charlie@example.com', 35, 'male')")

cur.execute("INSERT OR IGNORE INTO sales (id, product, amount, date, user_id) VALUES (1, 'Widget A', 100.0, '2023-01-01', 1)")
cur.execute("INSERT OR IGNORE INTO sales (id, product, amount, date, user_id) VALUES (2, 'Widget B', 150.0, '2023-01-02', 2)")
cur.execute("INSERT OR IGNORE INTO sales (id, product, amount, date, user_id) VALUES (3, 'Widget A', 200.0, '2023-01-03', 3)")

conn.commit()
conn.close()

print("Sample SQLite database created at backend/db/nl2q.db")
