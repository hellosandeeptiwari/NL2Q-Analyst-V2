"""
Test Azure SQL Database Connection
This script tests the database connection with detailed diagnostics
"""
import pyodbc
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_connection():
    """Test Azure SQL connection with detailed error reporting"""
    
    # Connection parameters
    config = {
        'host': os.getenv('AZURE_SQL_HOST', 'odsproduction.database.windows.net'),
        'port': int(os.getenv('AZURE_SQL_PORT', 1433)),
        'user': os.getenv('AZURE_SQL_USER', 'odsjobsuser'),
        'password': os.getenv('AZURE_SQL_PASSWORD', 'your_password_here'),
        'dbname': os.getenv('AZURE_SQL_DATABASE', 'DWHPRODIBSA')
    }
    
    print("=" * 70)
    print("🔍 Azure SQL Database Connection Test")
    print("=" * 70)
    print(f"\n📋 Connection Details:")
    print(f"   Server: {config['host']}")
    print(f"   Port: {config['port']}")
    print(f"   Database: {config['dbname']}")
    print(f"   User: {config['user']}")
    print(f"   Password: {'*' * len(config['password']) if config['password'] else 'NOT SET'}")
    
    # Check available ODBC drivers
    print(f"\n🔌 Available ODBC Drivers:")
    drivers = pyodbc.drivers()
    for driver in drivers:
        if 'SQL Server' in driver:
            print(f"   ✓ {driver}")
    
    # Try connection
    print(f"\n⏳ Attempting connection (30 second timeout)...")
    
    connection_string = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={config['host']},{config['port']};"
        f"DATABASE={config['dbname']};"
        f"UID={config['user']};"
        f"PWD={config['password']};"
        f"Encrypt=yes;TrustServerCertificate=no;"
        f"Connection Timeout=30;"
    )
    
    try:
        conn = pyodbc.connect(connection_string)
        print("\n✅ CONNECTION SUCCESSFUL! 🎉")
        print(f"   Connected to: {config['host']}")
        
        # Test query
        cursor = conn.cursor()
        cursor.execute("SELECT @@VERSION as version, DB_NAME() as database_name")
        row = cursor.fetchone()
        
        print(f"\n📊 Database Info:")
        print(f"   Database: {row.database_name}")
        print(f"   SQL Server Version: {row.version[:80]}...")
        
        # Close connection
        cursor.close()
        conn.close()
        print(f"\n✅ Connection closed successfully")
        print("=" * 70)
        return True
        
    except pyodbc.Error as e:
        error_code = e.args[0] if e.args else 'Unknown'
        error_msg = e.args[1] if len(e.args) > 1 else str(e)
        
        print(f"\n❌ CONNECTION FAILED!")
        print(f"   Error Code: {error_code}")
        print(f"   Error Message: {error_msg}")
        
        # Provide troubleshooting tips
        print(f"\n💡 Troubleshooting Steps:")
        
        if 'IM002' in str(error_code):
            print("   1. ODBC Driver not found - Install 'ODBC Driver 18 for SQL Server'")
        elif '08001' in str(error_code) or 'timeout' in error_msg.lower():
            print("   1. Network/Firewall Issue:")
            print("      - Add your IP (20.102.97.247) to Azure SQL firewall rules")
            print("      - Check if port 1433 is open in your network")
            print("      - Verify VPN/proxy settings")
        elif '28000' in str(error_code) or 'login' in error_msg.lower():
            print("   1. Authentication Issue:")
            print("      - Verify username and password")
            print("      - Check if user has access to the database")
        
        print("   2. Verify server address in Azure Portal")
        print("   3. Check if database is paused or offline")
        print("   4. Test from Azure Portal Query Editor")
        
        print("=" * 70)
        return False

if __name__ == "__main__":
    test_connection()
