import os
from dotenv import load_dotenv
import sys

# Load environment variables from .env file
load_dotenv()

print("=== Snowflake Connection Diagnostics ===")
print("Checking environment variables...")

# Required Snowflake parameters
required_params = [
    "SNOWFLAKE_USER", 
    "SNOWFLAKE_PASSWORD", 
    "SNOWFLAKE_ACCOUNT", 
    "SNOWFLAKE_WAREHOUSE", 
    "SNOWFLAKE_DATABASE", 
    "SNOWFLAKE_SCHEMA",
    "SNOWFLAKE_ROLE"
]

# Check if required parameters exist
missing_params = []
for param in required_params:
    value = os.getenv(param)
    if not value:
        missing_params.append(param)
        print(f"❌ Missing {param}")
    else:
        masked_value = value[:3] + "***" if param == "SNOWFLAKE_PASSWORD" else value
        print(f"✅ {param}: {masked_value}")

if missing_params:
    print(f"\n❌ Error: Missing required parameters: {', '.join(missing_params)}")
    print("Please update your .env file with the missing parameters.")
    sys.exit(1)

# Try importing Snowflake connector
print("\nChecking Snowflake connector...")
try:
    import snowflake.connector
    print("✅ Snowflake connector imported successfully")
except ImportError as e:
    print(f"❌ Error importing Snowflake connector: {e}")
    print("Try installing it with: pip install snowflake-connector-python")
    sys.exit(1)

# Try connecting to Snowflake
print("\nAttempting to connect to Snowflake...")
try:
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE")
    )
    
    print("✅ Connected to Snowflake successfully!")
    
    # Check database
    print("\nVerifying database access...")
    cursor = conn.cursor()
    cursor.execute("SELECT current_database(), current_schema(), current_warehouse(), current_role()")
    result = cursor.fetchone()
    print(f"Connected to Database: {result[0]}")
    print(f"Schema: {result[1]}")
    print(f"Warehouse: {result[2]}")
    print(f"Role: {result[3]}")
    
    # List tables
    print("\nListing tables...")
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    if tables:
        print(f"Found {len(tables)} tables:")
        for i, table in enumerate(tables[:5]):  # Show first 5 tables
            print(f"  - {table[1]}")  # table[1] is the table name
        if len(tables) > 5:
            print(f"  ... and {len(tables) - 5} more")
    else:
        print("No tables found in the current schema.")
    
    # Close connection
    conn.close()
    print("\n✅ Diagnostics completed successfully!")
    
except Exception as e:
    print(f"\n❌ Snowflake connection error: {e}")
    print("\nPossible solutions:")
    print("1. Verify your account identifier (it might need region prefix, e.g., 'us-east-1.account')")
    print("2. Ensure your username and password are correct")
    print("3. Check if the warehouse, database, and schema exist")
    print("4. Verify your network can reach Snowflake (no firewalls blocking it)")
    print("5. Check if your Snowflake account is active and not suspended")
    sys.exit(1)
