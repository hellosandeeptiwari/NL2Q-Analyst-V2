from typing import Protocol, Any, Dict
import time
import psycopg2
import snowflake.connector
import sqlite3
import pyodbc
import threading
import os
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

class RunResult:
    def __init__(self, rows, execution_time, error=None, plotly_spec=None, columns=None):
        self.rows = rows
        self.execution_time = execution_time
        self.error = error
        self.plotly_spec = plotly_spec
        self.columns = columns

class DBAdapter(Protocol):
    def connect(self) -> None: ...
    def health(self) -> dict: ...
    def run(self, sql: str, dry_run: bool=False) -> RunResult: ...
    def get_schema_snapshot(self, allowlist: list[str]) -> dict: ...

class PostgresAdapter:
    def __init__(self, config):
        self.config = config
        self.conn = None
        self.last_ok_query_at = None
        self.last_error_at = None
        self.last_error_code = None
        self.latency_ms = None
        self.lock = threading.Lock()

    def connect(self):
        self.conn = psycopg2.connect(**self.config)

    def health(self):
        try:
            start = time.time()
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
            self.latency_ms = int((time.time() - start) * 1000)
            self.last_ok_query_at = time.time()
            return {
                "connected": True,
                "latency_ms": self.latency_ms,
                "last_ok_query_at": self.last_ok_query_at,
                "last_error_at": self.last_error_at,
                "last_error_code": self.last_error_code
            }
        except Exception as e:
            self.last_error_at = time.time()
            self.last_error_code = str(e)
            return {
                "connected": False,
                "latency_ms": None,
                "last_ok_query_at": self.last_ok_query_at,
                "last_error_at": self.last_error_at,
                "last_error_code": self.last_error_code
            }

    def run(self, sql: str, dry_run: bool=False) -> RunResult:
        with self.lock:
            try:
                start = time.time()
                with self.conn.cursor() as cur:
                    if dry_run:
                        cur.execute(f"EXPLAIN {sql}")
                        rows = cur.fetchall()
                        columns = [desc[0] for desc in cur.description] if cur.description else []
                    else:
                        cur.execute(sql)
                        rows = cur.fetchall()
                        columns = [desc[0] for desc in cur.description] if cur.description else []
                execution_time = time.time() - start
                return RunResult(rows, execution_time, columns=columns)
            except Exception as e:
                return RunResult([], 0, error=str(e))

    def get_schema_snapshot(self, allowlist: list[str]) -> dict:
        with self.conn.cursor() as cur:
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            tables = [row[0] for row in cur.fetchall()]
            schema = {}
            for table in tables:
                cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}'")
                columns = cur.fetchall()
                schema[table] = {col[0]: col[1] for col in columns}
            return schema

    async def test_connection(self):
        """Test database connection"""
        health_result = self.health()
        if not health_result["connected"]:
            raise Exception(health_result["last_error_code"] or "Connection failed")
        return health_result

    async def get_table_names(self):
        """Get list of all table names"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT table_name, table_schema FROM information_schema.tables WHERE table_schema = 'public'")
            results = cur.fetchall()
            return [{"name": row[0], "schema": row[1]} for row in results]

    async def get_table_schema(self, table_name: str, schema_name: str = "public"):
        """Get column information for a specific table"""
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = '{table_name}' AND table_schema = '{schema_name}'")
            results = cur.fetchall()
            return [
                {
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2] == "YES"
                }
                for row in results
            ]

    async def execute_query(self, sql: str):
        """Execute a SQL query and return results"""
        with self.conn.cursor() as cur:
            cur.execute(sql)
            results = cur.fetchall()
            if results:
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in results]
            return []

class SnowflakeAdapter:
    def __init__(self, config):
        self.config = config
        self.conn = None
        self.last_ok_query_at = None
        self.last_error_at = None
        self.last_error_code = None
        self.latency_ms = None
        self.lock = threading.Lock()

    def connect(self):
        self.conn = snowflake.connector.connect(**self.config)
        # Try to set the database, schema, and role context
        try:
            cur = self.conn.cursor()
            
            # Set role first if specified
            if 'role' in self.config and self.config['role']:
                cur.execute(f"USE ROLE {self.config['role']}")
            
            # Set database context
            if 'database' in self.config and self.config['database']:
                cur.execute(f"USE DATABASE {self.config['database']}")
                
                # Set schema context
                if 'schema' in self.config and self.config['schema']:
                    cur.execute(f"USE SCHEMA {self.config['schema']}")
            
            # CRITICAL: Set quoted identifiers to FALSE for proper name handling
            cur.execute("ALTER SESSION SET QUOTED_IDENTIFIERS_IGNORE_CASE = FALSE")
            
            # Only print once when connection is established
            if not hasattr(self, '_context_set'):
                # Reduced logging to prevent spam in terminal
                # print(f"✅ Snowflake context: {self.config.get('role', 'default')}/{self.config.get('database', 'default')}/{self.config.get('schema', 'default')}")
                self._context_set = True
            
            cur.close()
        except Exception as e:
            # Log but don't fail - we can still work without setting the database context
            print(f"Warning: Could not set database context: {e}")
            print("Continuing with default database context...")

    def health(self):
        try:
            start = time.time()
            cur = self.conn.cursor()
            cur.execute("SELECT 1")
            self.latency_ms = int((time.time() - start) * 1000)
            self.last_ok_query_at = time.time()
            return {
                "connected": True,
                "latency_ms": self.latency_ms,
                "last_ok_query_at": self.last_ok_query_at,
                "last_error_at": self.last_error_at,
                "last_error_code": self.last_error_code
            }
        except Exception as e:
            self.last_error_at = time.time()
            self.last_error_code = str(e)
            return {
                "connected": False,
                "latency_ms": None,
                "last_ok_query_at": self.last_ok_query_at,
                "last_error_at": self.last_error_at,
                "last_error_code": self.last_error_code
            }

    def run(self, sql: str, dry_run: bool=False) -> RunResult:
        with self.lock:
            try:
                start = time.time()
                cur = self.conn.cursor()
                
                # Ensure quoted identifiers are handled properly for each query
                cur.execute("ALTER SESSION SET QUOTED_IDENTIFIERS_IGNORE_CASE = FALSE")
                
                if dry_run:
                    cur.execute(f"EXPLAIN {sql}")
                    rows = cur.fetchall()
                    columns = [desc[0] for desc in cur.description] if cur.description else []
                else:
                    cur.execute(sql)
                    rows = cur.fetchall()
                    columns = [desc[0] for desc in cur.description] if cur.description else []
                
                execution_time = time.time() - start
                cur.close()  # Close the cursor to prevent connection issues
                return RunResult(rows, execution_time, columns=columns)
            except Exception as e:
                return RunResult([], 0, error=str(e))

    def get_schema_snapshot(self, allowlist: list[str]) -> dict:
        cur = self.conn.cursor()
        schema = {}
        
        try:
            # First try to get tables from current context
            query = """
            SELECT TABLE_NAME, TABLE_SCHEMA
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
            LIMIT 50
            """
            cur.execute(query)
            tables = cur.fetchall()
            
            # If no tables found, try to get from any accessible schema
            if not tables:
                query = """
                SELECT TABLE_NAME, TABLE_SCHEMA
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA IS NOT NULL
                LIMIT 50
                """
                cur.execute(query)
                tables = cur.fetchall()
                
        except Exception as e:
            print(f"Warning: Could not fetch table list: {e}")
            # Return a mock schema for testing
            return {
                "SAMPLE_TABLE": {
                    "ID": "NUMBER",
                    "NAME": "VARCHAR",
                    "CREATED_DATE": "TIMESTAMP"
                }
            }
        
        for table_name, table_schema in tables[:10]:  # Limit to first 10 tables
            try:
                # Use schema-qualified table name
                full_table_name = f"{table_schema}.{table_name}"
                cur.execute(f"DESC TABLE {full_table_name}")
                columns = cur.fetchall()
                schema[table_name] = {col[0]: col[1] for col in columns}
            except Exception as e:
                print(f"Warning: Failed to describe table {table_name}: {e}")
                continue
                
        return schema

    async def test_connection(self):
        """Test database connection"""
        health_result = self.health()
        if not health_result["connected"]:
            raise Exception(health_result["last_error_code"] or "Connection failed")
        return health_result

    async def get_table_names(self):
        """Get list of all table names"""
        cur = self.conn.cursor()
        try:
            # Try to get tables from current context
            cur.execute("""
                SELECT TABLE_NAME, TABLE_SCHEMA
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
                ORDER BY TABLE_NAME
                LIMIT 50
            """)
            results = cur.fetchall()
            
            # If no tables found, try to get from any accessible schema  
            if not results:
                cur.execute("""
                    SELECT TABLE_NAME, TABLE_SCHEMA
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_SCHEMA IS NOT NULL
                    ORDER BY TABLE_NAME
                    LIMIT 50
                """)
                results = cur.fetchall()
                
            return [{"name": row[0], "schema": row[1]} for row in results]
        except Exception as e:
            print(f"Warning: Could not fetch table list: {e}")
            return []
        finally:
            cur.close()

    async def get_table_schema(self, table_name: str, schema_name: str = None):
        """Get column information for a specific table"""
        cur = self.conn.cursor()
        try:
            # Use schema-qualified table name
            if schema_name:
                full_table_name = f"{schema_name}.{table_name}"
            else:
                full_table_name = table_name
                
            cur.execute(f"DESC TABLE {full_table_name}")
            results = cur.fetchall()
            return [
                {
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2] == "Y"
                }
                for row in results
            ]
        except Exception as e:
            print(f"Warning: Failed to describe table {table_name}: {e}")
            return []
        finally:
            cur.close()

    async def execute_query(self, sql: str):
        """Execute a SQL query and return results"""
        cur = self.conn.cursor()
        try:
            cur.execute(sql)
            results = cur.fetchall()
            if results:
                columns = [desc[0] for desc in cur.description] if cur.description else []
                return [dict(zip(columns, row)) for row in results]
            return []
        finally:
            cur.close()


class SQLiteAdapter:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.last_ok_query_at = None
        self.last_error_at = None
        self.last_error_code = None
        self.latency_ms = None
        self.lock = threading.Lock()

    def connect(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

    def health(self):
        try:
            start = time.time()
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
            self.latency_ms = int((time.time() - start) * 1000)
            self.last_ok_query_at = time.time()
            return {
                "connected": True,
                "latency_ms": self.latency_ms,
                "last_ok_query_at": self.last_ok_query_at,
                "last_error_at": self.last_error_at,
                "last_error_code": self.last_error_code
            }
        except Exception as e:
            self.last_error_at = time.time()
            self.last_error_code = str(e)
            return {
                "connected": False,
                "latency_ms": None,
                "last_ok_query_at": self.last_ok_query_at,
                "last_error_at": self.last_error_at,
                "last_error_code": self.last_error_code
            }

    def run(self, sql: str, dry_run: bool = False) -> RunResult:
        with self.lock:
            try:
                start = time.time()
                cur = self.conn.cursor()
                cur.execute(sql)
                if not dry_run:
                    rows = cur.fetchall()
                else:
                    rows = []
                execution_time = time.time() - start
                self.conn.commit()
                return RunResult(rows, execution_time)
            except Exception as e:
                return RunResult([], 0, str(e))

    def get_schema_snapshot(self, allowlist: list[str]) -> dict:
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cur.fetchall()]
        schema = {}
        for table in tables:
            cur.execute(f"PRAGMA table_info({table})")
            columns = cur.fetchall()
            schema[table] = {col[1]: col[2] for col in columns}  # col[1] is name, col[2] is type
        return schema

    async def test_connection(self):
        """Test database connection"""
        health_result = self.health()
        if not health_result["connected"]:
            raise Exception(health_result["last_error_code"] or "Connection failed")
        return health_result

    async def get_table_names(self):
        """Get list of all table names"""
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        results = cur.fetchall()
        return [{"name": row[0], "schema": "main"} for row in results]

    async def get_table_schema(self, table_name: str, schema_name: str = "main"):
        """Get column information for a specific table"""
        cur = self.conn.cursor()
        cur.execute(f"PRAGMA table_info({table_name})")
        results = cur.fetchall()
        return [
            {
                "name": row[1],  # col[1] is name
                "type": row[2],  # col[2] is type
                "nullable": row[3] == 0  # col[3] is notnull (0 = nullable, 1 = not null)
            }
            for row in results
        ]

    async def execute_query(self, sql: str):
        """Execute a SQL query and return results"""
        cur = self.conn.cursor()
        cur.execute(sql)
        results = cur.fetchall()
        if results:
            columns = [desc[0] for desc in cur.description] if cur.description else []
            return [dict(zip(columns, row)) for row in results]
        return []


class AzureSQLAdapter:
    def __init__(self, config):
        self.config = config
        self.conn = None
        self.last_ok_query_at = None
        self.last_error_at = None
        self.last_error_code = None
        self.latency_ms = None
        self.lock = threading.Lock()

    def connect(self):
        # Debug config values
        print(f"🔍 Debug - Config values:")
        for key, value in self.config.items():
            if key == 'password':
                print(f"  {key}: {'*' * len(str(value)) if value else 'None'}")
            else:
                print(f"  {key}: {value}")
        
        try:
            # Build Azure SQL connection string with better timeout and retry settings
            connection_string = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.config['host']},{self.config['port']};"
                f"DATABASE={self.config['dbname']};"
                f"UID={self.config['user']};"
                f"PWD={self.config['password']};"
                f"Encrypt=yes;TrustServerCertificate=no;"
                f"Connection Timeout=10;"  # Reduced timeout to fail faster
                f"Login Timeout=10;"       # Login timeout
                f"ConnectRetryCount=3;"    # Retry attempts
                f"ConnectRetryInterval=10;" # Retry interval
            )
            
            print(f"🔗 Connecting to: {self.config['host']} database: {self.config['dbname']}")
            print("⏳ Attempting connection (10 second timeout)...")
            
            import time
            start_time = time.time()
            
            self.conn = pyodbc.connect(connection_string)
            
            connection_time = time.time() - start_time
            print(f"✅ Connected successfully in {connection_time:.2f} seconds")
            
            # Test the connection immediately
            print("🧪 Testing connection with simple query...")
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1 as test")
                result = cur.fetchone()
                print(f"✅ Connection test successful: {result}")
                
        except pyodbc.Error as e:
            error_code = e.args[0] if e.args else "Unknown"
            error_msg = e.args[1] if len(e.args) > 1 else str(e)
            
            print(f"❌ Azure SQL Connection Failed!")
            print(f"   Error Code: {error_code}")
            print(f"   Error Message: {error_msg}")
            
            # Provide specific troubleshooting based on error
            if "timeout" in error_msg.lower():
                print("🔥 TIMEOUT ERROR - Possible causes:")
                print("   1. Firewall blocking connection")
                print("   2. VPN/Network issues")
                print("   3. Server overloaded")
                print("   💡 Try: Check Azure SQL firewall rules for your IP")
            
            elif "login failed" in error_msg.lower():
                print("🔥 LOGIN ERROR - Possible causes:")
                print("   1. Wrong username/password")
                print("   2. User doesn't have database access")
                print("   💡 Try: Verify credentials in Azure portal")
                
            elif "server not found" in error_msg.lower():
                print("🔥 SERVER NOT FOUND - Possible causes:")
                print("   1. Wrong server name")
                print("   2. DNS resolution issues")
                print("   💡 Try: Ping the server hostname")
            
            raise Exception(f"Azure SQL connection failed: {error_msg}")
            
        except Exception as e:
            print(f"❌ Unexpected connection error: {str(e)}")
            raise

    def health(self):
        try:
            start = time.time()
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
            self.latency_ms = int((time.time() - start) * 1000)
            self.last_ok_query_at = time.time()
            return {
                "connected": True,
                "latency_ms": self.latency_ms,
                "last_ok_query_at": self.last_ok_query_at,
                "last_error_at": self.last_error_at,
                "last_error_code": self.last_error_code
            }
        except Exception as e:
            self.last_error_at = time.time()
            self.last_error_code = str(e)
            return {
                "connected": False,
                "latency_ms": None,
                "last_ok_query_at": self.last_ok_query_at,
                "last_error_at": self.last_error_at,
                "last_error_code": self.last_error_code
            }

    def run(self, sql: str, dry_run: bool = False) -> RunResult:
        with self.lock:
            try:
                start = time.time()
                cur = self.conn.cursor()
                cur.execute(sql)
                if not dry_run:
                    rows = cur.fetchall()
                    # Convert pyodbc Row objects to regular tuples
                    rows = [tuple(row) for row in rows]
                else:
                    rows = []
                execution_time = time.time() - start
                self.conn.commit()
                return RunResult(rows, execution_time)
            except Exception as e:
                return RunResult([], 0, str(e))

    def get_schema_snapshot(self, allowlist: list[str]) -> dict:
        cur = self.conn.cursor()
        
        print("🔍 DEBUG: Getting REAL schema from Azure SQL database...")
        
        cur.execute("""
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE 
            FROM INFORMATION_SCHEMA.COLUMNS 
            ORDER BY TABLE_NAME, ORDINAL_POSITION
        """)
        results = cur.fetchall()
        schema = {}
        
        for row in results:
            table_name, column_name, data_type = row
            if table_name not in schema:
                schema[table_name] = {}
            schema[table_name][column_name] = data_type
        
        # Debug output for the problematic tables
        for table_name in ['Reporting_BI_PrescriberProfile', 'Reporting_BI_PrescriberOverview', 'Reporting_BI_NGD']:
            if table_name in schema:
                columns = list(schema[table_name].keys())
                print(f"🔍 DEBUG: Real {table_name} columns: {columns[:10]}...")
                
                # Check specifically for product-related columns
                product_cols = [col for col in columns if 'product' in col.lower()]
                print(f"🔍 DEBUG: {table_name} product columns: {product_cols}")
        
        return schema

    async def test_connection(self):
        """Test database connection"""
        health_result = self.health()
        if not health_result["connected"]:
            raise Exception(health_result["last_error_code"] or "Connection failed")
        return health_result

    async def get_table_names(self):
        """Get list of all table names"""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT TABLE_NAME, TABLE_SCHEMA 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
        """)
        results = cur.fetchall()
        return [{"name": row[0], "schema": row[1]} for row in results]

    async def get_table_schema(self, table_name: str, schema_name: str = "dbo"):
        """Get column information for a specific table"""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = ? AND TABLE_SCHEMA = ?
            ORDER BY ORDINAL_POSITION
        """, (table_name, schema_name))
        results = cur.fetchall()
        return [
            {
                "name": row[0],
                "type": row[1],
                "nullable": row[2] == "YES"
            }
            for row in results
        ]

    async def execute_query(self, sql: str):
        """Execute a SQL query and return results"""
        cur = self.conn.cursor()
        cur.execute(sql)
        results = cur.fetchall()
        if results:
            # Convert to list of dictionaries
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in results]
        return []
    
    def get_real_table_columns(self, table_name: str, schema_name: str = "dbo") -> list:
        """Get the ACTUAL column names from the database for a specific table"""
        cur = self.conn.cursor()
        try:
            print(f"🔍 DEBUG: Getting REAL columns for {schema_name}.{table_name}")
            cur.execute("""
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = ? AND TABLE_SCHEMA = ?
                ORDER BY ORDINAL_POSITION
            """, (table_name, schema_name))
            results = cur.fetchall()
            
            columns = []
            for row in results:
                columns.append({
                    'column_name': row[0],
                    'data_type': row[1],
                    'is_nullable': row[2] == 'YES'
                })
            
            print(f"🔍 DEBUG: Found {len(columns)} real columns for {table_name}")
            if columns:
                col_names = [col['column_name'] for col in columns]
                print(f"🔍 DEBUG: Column names: {col_names[:10]}...")
                
                # Look for product-related columns specifically
                product_cols = [col for col in col_names if 'product' in col.lower() or 'tirosint' in col.lower()]
                if product_cols:
                    print(f"🔍 DEBUG: Product-related columns: {product_cols}")
                else:
                    print(f"🔍 DEBUG: No product columns found, checking target flags...")
                    target_cols = [col for col in col_names if 'target' in col.lower() or 'flag' in col.lower()]
                    print(f"🔍 DEBUG: Target/Flag columns: {target_cols}")
            
            return columns
            
        except Exception as e:
            print(f"❌ Error getting real columns for {table_name}: {e}")
            return []


def get_adapter(db_engine=None):
    # Load config from .env or config file
    import os
    if db_engine is None:
        db_engine = os.getenv("DB_ENGINE", "sqlite")
    if db_engine == "postgres":
        config = {
            "host": os.getenv("PG_HOST"),
            "port": os.getenv("PG_PORT"),
            "user": os.getenv("PG_USER"),
            "password": os.getenv("PG_PASSWORD"),
            "dbname": os.getenv("PG_DBNAME")
        }
        adapter = PostgresAdapter(config)
        adapter.connect()
        return adapter
    elif db_engine == "snowflake":
        config = {
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA"),
            "role": os.getenv("SNOWFLAKE_ROLE")
        }
        adapter = SnowflakeAdapter(config)
        adapter.connect()
        return adapter
    elif db_engine == "azure_sql":
        # Direct Azure SQL connection - no fallbacks
        config = {
            "host": os.getenv("AZURE_SQL_HOST"),
            "port": os.getenv("AZURE_SQL_PORT", 1433),
            "user": os.getenv("AZURE_SQL_USER"),
            "password": os.getenv("AZURE_SQL_PASSWORD"),
            "dbname": os.getenv("AZURE_SQL_DATABASE")  # Fixed: use AZURE_SQL_DATABASE not AZURE_SQL_DBNAME
        }
        print("✅ Connecting to Azure SQL Server directly...")
        adapter = AzureSQLAdapter(config)
        adapter.connect()
        return adapter
    elif db_engine == "sqlite":
        db_path = os.getenv("SQLITE_DB_PATH", "backend/db/nl2q.db")
        adapter = SQLiteAdapter(db_path)
        adapter.connect()
        return adapter
    else:
        raise ValueError("Unsupported DB_ENGINE")
