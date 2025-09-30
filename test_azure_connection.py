#!/usr/bin/env python3
"""
Quick Azure SQL Connection Test
Helps diagnose connection issues with detailed error reporting
"""

import os
import time
import socket
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_network_connectivity():
    """Test basic network connectivity to Azure SQL server"""
    host = os.getenv("AZURE_SQL_HOST")
    port = int(os.getenv("AZURE_SQL_PORT", 1433))
    
    print(f"🌐 Testing network connectivity to {host}:{port}")
    
    try:
        start_time = time.time()
        sock = socket.create_connection((host, port), timeout=10)
        sock.close()
        connection_time = time.time() - start_time
        print(f"✅ Network connection successful in {connection_time:.2f} seconds")
        return True
    except socket.timeout:
        print("❌ Network connection TIMEOUT - Server unreachable or firewall blocking")
        return False
    except socket.gaierror as e:
        print(f"❌ DNS resolution failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Network connection failed: {e}")
        return False

def test_azure_sql_connection():
    """Test full Azure SQL database connection"""
    try:
        import pyodbc
        print("✅ pyodbc driver available")
    except ImportError:
        print("❌ pyodbc not installed. Run: pip install pyodbc")
        return False
    
    # Check required environment variables
    required_vars = ["AZURE_SQL_HOST", "AZURE_SQL_USER", "AZURE_SQL_PASSWORD", "AZURE_SQL_DATABASE"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return False
    
    config = {
        "host": os.getenv("AZURE_SQL_HOST"),
        "port": os.getenv("AZURE_SQL_PORT", 1433),
        "user": os.getenv("AZURE_SQL_USER"),
        "password": os.getenv("AZURE_SQL_PASSWORD"),
        "dbname": os.getenv("AZURE_SQL_DATABASE")
    }
    
    print("🔍 Configuration:")
    for key, value in config.items():
        if key == 'password':
            print(f"  {key}: {'*' * len(str(value)) if value else 'None'}")
        else:
            print(f"  {key}: {value}")
    
    # Test network connectivity first
    if not test_network_connectivity():
        print("\n💡 Network connectivity failed. Possible solutions:")
        print("   1. Add your IP to Azure SQL firewall rules")
        print("   2. Check if you're behind a corporate firewall")
        print("   3. Verify VPN connection if required")
        return False
    
    # Test database connection
    print(f"\n🔗 Testing Azure SQL database connection...")
    
    try:
        connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={config['host']},{config['port']};"
            f"DATABASE={config['dbname']};"
            f"UID={config['user']};"
            f"PWD={config['password']};"
            f"Encrypt=yes;TrustServerCertificate=no;"
            f"Connection Timeout=10;"
            f"Login Timeout=10;"
        )
        
        start_time = time.time()
        conn = pyodbc.connect(connection_string)
        connection_time = time.time() - start_time
        
        print(f"✅ Database connection successful in {connection_time:.2f} seconds")
        
        # Test a simple query
        with conn.cursor() as cur:
            cur.execute("SELECT @@VERSION as version, DB_NAME() as database_name")
            result = cur.fetchone()
            print(f"✅ Query test successful:")
            print(f"   Database: {result[1]}")
            print(f"   Version: {result[0][:100]}...")
        
        conn.close()
        return True
        
    except pyodbc.Error as e:
        error_code = e.args[0] if e.args else "Unknown"
        error_msg = e.args[1] if len(e.args) > 1 else str(e)
        
        print(f"❌ Database connection failed:")
        print(f"   Error Code: {error_code}")
        print(f"   Error Message: {error_msg}")
        
        # Provide specific troubleshooting
        if "28000" in error_code:  # Login failed
            print("\n💡 Login failed. Check:")
            print("   1. Username and password are correct")
            print("   2. User has access to the specific database")
            print("   3. Account is not locked or expired")
            
        elif "08001" in error_code or "timeout" in error_msg.lower():
            print("\n💡 Connection timeout. Check:")
            print("   1. Azure SQL firewall allows your IP address")
            print("   2. Network/VPN connectivity")
            print("   3. Server is running and accessible")
            
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Azure SQL Connection Diagnostic Tool")
    print("=" * 50)
    
    success = test_azure_sql_connection()
    
    if success:
        print("\n🎉 All tests passed! Azure SQL connection is working.")
    else:
        print("\n❌ Connection tests failed. Check the errors above.")
        print("\n🔧 Common solutions:")
        print("   1. Add your IP to Azure SQL Server firewall rules")
        print("   2. Verify environment variables in .env file")
        print("   3. Check Azure SQL server is running")
        print("   4. Ensure user has proper database permissions")