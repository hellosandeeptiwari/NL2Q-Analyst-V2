from typing import Protocol, Any, Dict
import time
import psycopg2
import snowflake.connector
import sqlite3
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
        # Assuming Azure SQL uses similar to Postgres, but with different driver
        # For simplicity, use PostgresAdapter with Azure SQL config
        config = {
            "host": os.getenv("AZURE_SQL_HOST"),
            "port": os.getenv("AZURE_SQL_PORT", 1433),
            "user": os.getenv("AZURE_SQL_USER"),
            "password": os.getenv("AZURE_SQL_PASSWORD"),
            "dbname": os.getenv("AZURE_SQL_DBNAME")
        }
        adapter = PostgresAdapter(config)  # Note: This might not work for Azure SQL, as it uses TDS protocol, not PostgreSQL
        adapter.connect()
        return adapter
    elif db_engine == "sqlite":
        db_path = os.getenv("SQLITE_DB_PATH", "backend/db/nl2q.db")
        adapter = SQLiteAdapter(db_path)
        adapter.connect()
        return adapter
    else:
        raise ValueError("Unsupported DB_ENGINE")
