"""
Quick SQLite Adapter Fix
"""

import sqlite3
import time

class FixedSQLiteAdapter:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.last_ok_query_at = None
        self.last_error_at = None
        self.last_error_code = None
        self.latency_ms = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

    def health(self):
        try:
            start = time.time()
            cur = self.conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
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

    def run(self, sql: str, dry_run: bool = False):
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
            cur.close()
            
            # Return in expected format
            class RunResult:
                def __init__(self, rows, execution_time, error=None):
                    self.rows = rows
                    self.execution_time = execution_time
                    self.error = error
            
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
        cur.close()
        return schema
