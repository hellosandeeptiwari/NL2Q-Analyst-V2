from typing import Protocol, Any, Dict
import time
import psycopg2
import snowflake.connector
import sqlite3
import threading

class RunResult:
    def __init__(self, rows, execution_time, error=None, plotly_spec=None):
        self.rows = rows
        self.execution_time = execution_time
        self.error = error
        self.plotly_spec = plotly_spec

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
                    else:
                        cur.execute(sql)
                        rows = cur.fetchall()
                execution_time = time.time() - start
                return RunResult(rows, execution_time)
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
                if dry_run:
                    cur.execute(f"EXPLAIN {sql}")
                    rows = cur.fetchall()
                else:
                    cur.execute(sql)
                    rows = cur.fetchall()
                execution_time = time.time() - start
                return RunResult(rows, execution_time)
            except Exception as e:
                return RunResult([], 0, error=str(e))

    def get_schema_snapshot(self, allowlist: list[str]) -> dict:
        cur = self.conn.cursor()
        cur.execute("SHOW TABLES")
        tables = [row[1] for row in cur.fetchall()]
        schema = {}
        for table in tables:
            # Use quoted identifiers for Snowflake case sensitivity and special characters
            quoted_table = f'"{table}"'
            cur.execute(f"DESC TABLE {quoted_table}")
            columns = cur.fetchall()
            schema[table] = {col[0]: col[1] for col in columns}
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
            "schema": os.getenv("SNOWFLAKE_SCHEMA")
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
