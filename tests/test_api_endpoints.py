import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import json
import os
from backend.main import app

# Test client
client = TestClient(app)

# Sample test data
SAMPLE_QUERY_DATA = {
    "natural_language": "Show me all customers from New York",
    "job_id": "test_job_123",
    "db_type": "sqlite"
}

SAMPLE_INSIGHTS_DATA = {
    "location": "test_location_123",
    "query": "Show me sales data"
}

SAMPLE_FAVORITES_DATA = {
    "query": "SELECT * FROM customers WHERE city = 'New York'"
}

# Mock authentication token
VALID_TOKEN = "Bearer test_token_123"

class TestHealthAPI:
    """Test cases for /health endpoint"""

    def test_health_success(self):
        """Test successful health check"""
        with patch('backend.main.adapter') as mock_adapter:
            mock_adapter.health.return_value = {
                "connected": True,
                "latency_ms": 15,
                "last_ok_query_at": 1234567890,
                "last_error_at": None,
                "last_error_code": None
            }

            response = client.get("/health", headers={"Authorization": VALID_TOKEN})
            assert response.status_code == 200
            data = response.json()
            assert data["connected"] is True
            assert data["latency_ms"] == 15

    def test_health_database_error(self):
        """Test health check with database error"""
        with patch('backend.main.adapter') as mock_adapter:
            mock_adapter.health.side_effect = Exception("Database connection failed")

            response = client.get("/health", headers={"Authorization": VALID_TOKEN})
            assert response.status_code == 500

class TestSchemaAPI:
    """Test cases for /schema endpoint"""

    def test_schema_success(self):
        """Test successful schema retrieval"""
        with patch('backend.main.schema_cache') as mock_cache:
            mock_cache.__getitem__ = Mock(return_value={
                "customers": {"id": "INTEGER", "name": "TEXT", "city": "TEXT"},
                "orders": {"id": "INTEGER", "customer_id": "INTEGER", "amount": "REAL"}
            })

            response = client.get("/schema", headers={"Authorization": VALID_TOKEN})
            assert response.status_code == 200
            data = response.json()
            assert "customers" in data
            assert "orders" in data

class TestQueryAPI:
    """Test cases for /query endpoint"""

    def test_query_success_sqlite(self):
        """Test successful query execution with SQLite"""
        with patch('backend.main.generate_sql') as mock_generate, \
             patch('backend.main.get_adapter') as mock_get_adapter, \
             patch('backend.main.storage') as mock_storage, \
             patch('backend.main.bias_detector') as mock_bias_detector:

            # Mock the SQL generation
            mock_sql_result = Mock()
            mock_sql_result.sql = "SELECT * FROM customers WHERE city = 'New York'"
            mock_sql_result.suggestions = ["Try filtering by state", "Add date range"]
            mock_generate.return_value = mock_sql_result

            # Mock the database adapter
            mock_adapter = Mock()
            mock_adapter.run.return_value = Mock(
                rows=[{"id": 1, "name": "John Doe", "city": "New York"}],
                execution_time=0.05,
                error=None
            )
            mock_get_adapter.return_value = mock_adapter

            # Mock storage
            mock_storage.save_data.return_value = "/tmp/test_data.json"

            # Mock bias detector
            mock_bias_detector.detect_bias.return_value = "No significant bias detected"

            response = client.post("/query",
                                 json=SAMPLE_QUERY_DATA,
                                 headers={"Authorization": VALID_TOKEN})

            assert response.status_code == 200
            data = response.json()
            assert "sql" in data
            assert "rows" in data
            assert "suggestions" in data
            assert "bias_report" in data
            assert len(data["rows"]) == 1

    def test_query_with_different_db_type(self):
        """Test query with different database type"""
        query_data = SAMPLE_QUERY_DATA.copy()
        query_data["db_type"] = "postgres"

        with patch('backend.main.generate_sql') as mock_generate, \
             patch('backend.main.get_adapter') as mock_get_adapter, \
             patch('backend.main.storage') as mock_storage, \
             patch('backend.main.bias_detector') as mock_bias_detector:

            mock_sql_result = Mock()
            mock_sql_result.sql = "SELECT * FROM customers"
            mock_sql_result.suggestions = []
            mock_generate.return_value = mock_sql_result

            mock_adapter = Mock()
            mock_adapter.run.return_value = Mock(rows=[], execution_time=0.02, error=None)
            mock_get_adapter.return_value = mock_adapter

            mock_storage.save_data.return_value = "/tmp/test_data.json"
            mock_bias_detector.detect_bias.return_value = "Analysis complete"

            response = client.post("/query",
                                 json=query_data,
                                 headers={"Authorization": VALID_TOKEN})

            assert response.status_code == 200
            # Verify get_adapter was called with postgres
            mock_get_adapter.assert_called_with("postgres")

    def test_query_sql_generation_error(self):
        """Test query with SQL generation error"""
        with patch('backend.main.generate_sql') as mock_generate:
            mock_generate.side_effect = Exception("Failed to generate SQL")

            response = client.post("/query",
                                 json=SAMPLE_QUERY_DATA,
                                 headers={"Authorization": VALID_TOKEN})

            assert response.status_code == 500

    def test_query_database_error(self):
        """Test query with database execution error"""
        with patch('backend.main.generate_sql') as mock_generate, \
             patch('backend.main.get_adapter') as mock_get_adapter:

            mock_sql_result = Mock()
            mock_sql_result.sql = "SELECT * FROM invalid_table"
            mock_sql_result.suggestions = []
            mock_generate.return_value = mock_sql_result

            mock_adapter = Mock()
            mock_adapter.run.side_effect = Exception("Table does not exist")
            mock_get_adapter.return_value = mock_adapter

            response = client.post("/query",
                                 json=SAMPLE_QUERY_DATA,
                                 headers={"Authorization": VALID_TOKEN})

            assert response.status_code == 500

    def test_query_missing_natural_language(self):
        """Test query with missing natural language input"""
        invalid_data = {"job_id": "test_job_123"}

        response = client.post("/query",
                             json=invalid_data,
                             headers={"Authorization": VALID_TOKEN})

        assert response.status_code == 500

class TestCSVExportAPI:
    """Test cases for /csv/{job_id} endpoint"""

    def test_csv_export_success(self):
        """Test successful CSV export"""
        with patch('backend.main.storage') as mock_storage, \
             patch('builtins.open', create=True) as mock_open:

            mock_storage.save_data.return_value = "/tmp/test_data.csv"
            mock_file = Mock()
            mock_open.return_value = mock_file

            response = client.get("/csv/test_job_123",
                                headers={"Authorization": VALID_TOKEN})

            assert response.status_code == 200
            mock_storage.save_data.assert_called_with([], "test_job_123")

    def test_csv_export_s3_url(self):
        """Test CSV export with S3 URL"""
        with patch('backend.main.storage') as mock_storage:
            mock_storage.save_data.return_value = "s3://bucket/test_data.csv"

            response = client.get("/csv/test_job_123",
                                headers={"Authorization": VALID_TOKEN})

            assert response.status_code == 200
            data = response.json()
            assert data["url"] == "s3://bucket/test_data.csv"

class TestInsightsAPI:
    """Test cases for /insights endpoint"""

    def test_insights_success(self):
        """Test successful insights generation"""
        with patch('backend.main.storage') as mock_storage:
            mock_storage.load_data.return_value = [
                {"sales": 1000, "region": "North"},
                {"sales": 1500, "region": "South"}
            ]
            mock_storage.generate_insights.return_value = "Sales are trending upward in the South region"

            response = client.post("/insights",
                                 json=SAMPLE_INSIGHTS_DATA,
                                 headers={"Authorization": VALID_TOKEN})

            assert response.status_code == 200
            data = response.json()
            assert "insight" in data
            assert "trending upward" in data["insight"]

    def test_insights_data_not_found(self):
        """Test insights with missing data"""
        with patch('backend.main.storage') as mock_storage:
            mock_storage.load_data.side_effect = FileNotFoundError("Data not found")

            response = client.post("/insights",
                                 json=SAMPLE_INSIGHTS_DATA,
                                 headers={"Authorization": VALID_TOKEN})

            assert response.status_code == 500

class TestHistoryAPI:
    """Test cases for /history endpoint"""

    def test_history_success(self):
        """Test successful history retrieval"""
        with patch('backend.main.get_recent_queries') as mock_get_history:
            mock_get_history.return_value = [
                {
                    "nl_query": "Show me all customers",
                    "sql": "SELECT * FROM customers",
                    "timestamp": "2025-09-01T10:00:00Z"
                },
                {
                    "nl_query": "Show me sales data",
                    "sql": "SELECT * FROM sales",
                    "timestamp": "2025-09-01T10:05:00Z"
                }
            ]

            response = client.get("/history", headers={"Authorization": VALID_TOKEN})

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["nl_query"] == "Show me all customers"

class TestAnalyticsAPI:
    """Test cases for /analytics endpoint"""

    def test_analytics_success(self):
        """Test successful analytics retrieval"""
        with patch('backend.analytics.usage.get_usage_stats') as mock_get_stats:
            mock_get_stats.return_value = {
                "total_queries": 150,
                "active_users": 25,
                "avg_response_time": 0.8,
                "error_rate": 0.02
            }

            response = client.get("/analytics", headers={"Authorization": VALID_TOKEN})

            assert response.status_code == 200
            data = response.json()
            assert data["total_queries"] == 150
            assert data["active_users"] == 25

class TestErrorsAPI:
    """Test cases for /errors endpoint"""

    def test_errors_success(self):
        """Test successful error reports retrieval"""
        with patch('backend.main.get_error_reports') as mock_get_errors:
            mock_get_errors.return_value = [
                {
                    "error": "Database connection timeout",
                    "timestamp": "2025-09-01T10:00:00Z",
                    "context": {"query": "SELECT * FROM large_table"}
                }
            ]

            response = client.get("/errors", headers={"Authorization": VALID_TOKEN})

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert "Database connection timeout" in data[0]["error"]

class TestLogsAPI:
    """Test cases for /logs endpoint"""

    def test_logs_success(self):
        """Test successful logs retrieval"""
        mock_log_data = [
            {
                "timestamp": "2025-09-01T10:00:00Z",
                "action": "query_executed",
                "user": "test_user",
                "details": {"sql": "SELECT * FROM customers"}
            }
        ]

        with patch('builtins.open') as mock_open:
            mock_file = Mock()
            mock_file.__iter__ = Mock(return_value=iter([json.dumps(mock_log_data[0])]))
            mock_open.return_value.__enter__.return_value = mock_file

            response = client.get("/logs", headers={"Authorization": VALID_TOKEN})

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["action"] == "query_executed"

class TestAuthentication:
    """Test cases for authentication"""

    def test_missing_token(self):
        """Test API access without authentication token"""
        response = client.get("/health")
        assert response.status_code == 401

    def test_invalid_token(self):
        """Test API access with invalid token"""
        response = client.get("/health", headers={"Authorization": "Bearer invalid_token"})
        assert response.status_code == 401

class TestSSEStatusAPI:
    """Test cases for /events/status SSE endpoint"""

    def test_sse_status_connection(self):
        """Test SSE status endpoint connection"""
        # Note: Testing SSE endpoints is complex, this is a basic connectivity test
        response = client.get("/events/status", headers={"Authorization": VALID_TOKEN})

        # SSE endpoints typically return 200 for connection establishment
        assert response.status_code in [200, 201]

# Integration test fixtures
@pytest.fixture
def sample_customer_data():
    """Sample customer data for testing"""
    return [
        {"id": 1, "name": "John Doe", "email": "john@example.com", "city": "New York"},
        {"id": 2, "name": "Jane Smith", "email": "jane@example.com", "city": "Los Angeles"},
        {"id": 3, "name": "Bob Johnson", "email": "bob@example.com", "city": "Chicago"}
    ]

@pytest.fixture
def sample_sales_data():
    """Sample sales data for testing"""
    return [
        {"id": 1, "customer_id": 1, "product": "Widget A", "amount": 99.99, "date": "2025-09-01"},
        {"id": 2, "customer_id": 2, "product": "Widget B", "amount": 149.99, "date": "2025-09-02"},
        {"id": 3, "customer_id": 1, "product": "Widget C", "amount": 79.99, "date": "2025-09-03"}
    ]

@pytest.fixture
def sample_inventory_data():
    """Sample inventory data for testing"""
    return [
        {"id": 1, "product_name": "Widget A", "stock_quantity": 150, "unit_price": 99.99},
        {"id": 2, "product_name": "Widget B", "stock_quantity": 75, "unit_price": 149.99},
        {"id": 3, "product_name": "Widget C", "stock_quantity": 200, "unit_price": 79.99}
    ]

# Performance test
def test_query_performance():
    """Test query performance under load"""
    import time

    with patch('backend.main.generate_sql') as mock_generate, \
         patch('backend.main.get_adapter') as mock_get_adapter, \
         patch('backend.main.storage') as mock_storage, \
         patch('backend.main.bias_detector') as mock_bias_detector:

        mock_sql_result = Mock()
        mock_sql_result.sql = "SELECT COUNT(*) FROM customers"
        mock_sql_result.suggestions = []
        mock_generate.return_value = mock_sql_result

        mock_adapter = Mock()
        mock_adapter.run.return_value = Mock(
            rows=[{"count": 1000}],
            execution_time=0.05,
            error=None
        )
        mock_get_adapter.return_value = mock_adapter

        mock_storage.save_data.return_value = "/tmp/test_data.json"
        mock_bias_detector.detect_bias.return_value = "No bias detected"

        start_time = time.time()

        # Execute multiple queries
        for i in range(10):
            response = client.post("/query",
                                 json={
                                     "natural_language": f"Count customers {i}",
                                     "job_id": f"perf_test_{i}"
                                 },
                                 headers={"Authorization": VALID_TOKEN})

            assert response.status_code == 200

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert total_time < 5.0  # 5 seconds for 10 queries

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
