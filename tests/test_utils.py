"""
Test utilities and helper functions for NL2Q Agent testing
"""

import json
import tempfile
import sqlite3
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime, timedelta
import random

def create_mock_db_connection():
    """Create a mock database connection for testing"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.commit.return_value = None
    mock_conn.close.return_value = None
    return mock_conn, mock_cursor

def create_test_database(schema: Dict[str, Dict], data: Dict[str, List[Dict]]):
    """Create an in-memory SQLite database with test data"""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Create tables
    for table_name, columns in schema.items():
        columns_str = ', '.join([f"{col} {dtype}" for col, dtype in columns.items()])
        cursor.execute(f"CREATE TABLE {table_name} ({columns_str})")

    # Insert test data
    for table_name, rows in data.items():
        if rows:
            columns = list(rows[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            columns_str = ', '.join(columns)

            for row in rows:
                values = [row[col] for col in columns]
                cursor.execute(f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})", values)

    conn.commit()
    return conn

def create_mock_openai_response(content: str, role: str = "assistant"):
    """Create a mock OpenAI API response"""
    return {
        "choices": [{
            "message": {
                "role": role,
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }

def create_mock_jwt_token(user_id: str = "test_user", role: str = "user"):
    """Create a mock JWT token for testing"""
    import jwt
    from datetime import datetime, timedelta

    payload = {
        "user_id": user_id,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.now()
    }

    # Use a test secret key
    token = jwt.encode(payload, "test_secret_key", algorithm="HS256")
    return token

def generate_random_data(table_name: str, num_rows: int = 10) -> List[Dict]:
    """Generate random test data for a given table"""
    if table_name == "customers":
        return [{
            "id": i,
            "first_name": f"First{i}",
            "last_name": f"Last{i}",
            "email": f"user{i}@example.com",
            "phone": f"+1-555-0{i:03d}",
            "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"][i % 5],
            "state": ["NY", "CA", "IL", "TX", "AZ"][i % 5],
            "country": "USA",
            "registration_date": f"2023-{(i%12)+1:02d}-{(i%28)+1:02d}",
            "is_active": random.choice([True, False]),
            "customer_type": random.choice(["Premium", "Standard", "Basic"])
        } for i in range(1, num_rows + 1)]

    elif table_name == "products":
        categories = ["Electronics", "Furniture", "Food & Beverage", "Sports & Fitness"]
        return [{
            "id": i,
            "name": f"Product {i}",
            "category": random.choice(categories),
            "price": round(random.uniform(10, 500), 2),
            "cost": round(random.uniform(5, 250), 2),
            "stock_quantity": random.randint(0, 200),
            "supplier": f"Supplier{random.randint(1, 5)}",
            "rating": round(random.uniform(1, 5), 1),
            "is_available": random.choice([True, False])
        } for i in range(1, num_rows + 1)]

    elif table_name == "orders":
        return [{
            "id": i,
            "customer_id": random.randint(1, 100),
            "order_date": f"2023-{(i%12)+1:02d}-{(i%28)+1:02d}",
            "total_amount": round(random.uniform(20, 1000), 2),
            "status": random.choice(["pending", "processing", "shipped", "completed", "cancelled"]),
            "payment_method": random.choice(["credit_card", "debit_card", "paypal", "bank_transfer"]),
            "shipping_address": f"{random.randint(100, 999)} Test St, Test City, TS {random.randint(10000, 99999)}"
        } for i in range(1, num_rows + 1)]

    return []

def create_mock_request_data(query: str = None, db_type: str = "sqlite", **kwargs):
    """Create mock request data for API testing"""
    data = {
        "query": query or "Show me all customers",
        "db_type": db_type,
        "db_config": {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "username": "test_user",
            "password": "test_pass"
        }
    }
    data.update(kwargs)
    return data

def create_mock_response_data(sql: str = None, results: List[Dict] = None, **kwargs):
    """Create mock response data for API testing"""
    data = {
        "sql": sql or "SELECT * FROM customers",
        "results": results or [{"id": 1, "name": "Test Customer"}],
        "execution_time": 0.5,
        "row_count": len(results) if results else 1
    }
    data.update(kwargs)
    return data

def assert_api_response(response, expected_status: int = 200, expected_keys: List[str] = None):
    """Assert common API response properties"""
    assert response.status_code == expected_status

    if expected_keys:
        response_data = response.get_json()
        for key in expected_keys:
            assert key in response_data, f"Expected key '{key}' not found in response"

def create_temp_file(content: str, suffix: str = ".txt") -> str:
    """Create a temporary file with given content"""
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(content)
        return f.name

def mock_openai_call(response_content: str):
    """Decorator to mock OpenAI API calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with patch('openai.ChatCompletion.create') as mock_create:
                mock_create.return_value = create_mock_openai_response(response_content)
                return func(*args, **kwargs)
        return wrapper
    return decorator

def generate_performance_test_data(size: str = "small") -> Dict[str, List[Dict]]:
    """Generate test data of different sizes for performance testing"""
    sizes = {
        "small": 100,
        "medium": 1000,
        "large": 10000
    }

    num_rows = sizes.get(size, 100)

    return {
        "customers": generate_random_data("customers", num_rows),
        "products": generate_random_data("products", num_rows // 2),
        "orders": generate_random_data("orders", num_rows * 2)
    }

def create_mock_analytics_data():
    """Create mock analytics data for testing"""
    return {
        "total_queries": 1250,
        "successful_queries": 1180,
        "failed_queries": 70,
        "average_response_time": 2.3,
        "popular_tables": ["customers", "orders", "products"],
        "query_trends": {
            "2023-09-01": 45,
            "2023-09-02": 52,
            "2023-09-03": 38
        },
        "error_types": {
            "syntax_error": 25,
            "table_not_found": 15,
            "column_not_found": 30
        }
    }

def create_mock_audit_log():
    """Create mock audit log entries for testing"""
    return [
        {
            "timestamp": "2023-09-01T10:00:00Z",
            "user_id": "user123",
            "action": "query",
            "query": "SELECT * FROM customers",
            "success": True,
            "execution_time": 1.2
        },
        {
            "timestamp": "2023-09-01T10:05:00Z",
            "user_id": "user456",
            "action": "query",
            "query": "SELECT * FROM invalid_table",
            "success": False,
            "error": "table not found",
            "execution_time": 0.1
        }
    ]

def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype: bool = True):
    """Assert that two pandas DataFrames are equal"""
    pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)

def create_mock_file_upload(filename: str = "test.csv", content: str = "id,name\n1,Test"):
    """Create a mock file upload for testing"""
    from werkzeug.datastructures import FileStorage
    from io import BytesIO

    file_data = BytesIO(content.encode('utf-8'))
    file = FileStorage(
        stream=file_data,
        filename=filename,
        content_type='text/csv'
    )
    return file

class TestTimer:
    """Context manager for timing test execution"""
    def __init__(self, name: str = "Test"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        print(f"Starting {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        print(".2f")

def create_mock_error_response(error_type: str, message: str):
    """Create mock error response for testing"""
    return {
        "error": {
            "type": error_type,
            "message": message,
            "timestamp": datetime.now().isoformat()
        },
        "status": "error"
    }

def validate_sql_query(sql: str) -> bool:
    """Basic SQL query validation for testing"""
    # Simple validation - check for basic SQL keywords
    sql_lower = sql.lower().strip()

    # Must contain SELECT, FROM, or other valid SQL commands
    valid_starts = ['select', 'insert', 'update', 'delete', 'create', 'drop', 'alter']
    return any(sql_lower.startswith(keyword) for keyword in valid_starts)

def create_mock_schema_cache():
    """Create mock schema cache for testing"""
    return {
        "customers": {
            "columns": ["id", "name", "email", "created_at"],
            "types": ["INTEGER", "TEXT", "TEXT", "DATETIME"],
            "primary_key": "id"
        },
        "products": {
            "columns": ["id", "name", "price", "category"],
            "types": ["INTEGER", "TEXT", "REAL", "TEXT"],
            "primary_key": "id"
        }
    }
