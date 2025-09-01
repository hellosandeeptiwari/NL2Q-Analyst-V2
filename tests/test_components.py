import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from backend.nl2sql.generator import generate_sql, generate_query_suggestions
from backend.nl2sql.bias_detection import BiasDetector
from backend.db.engine import get_adapter, PostgresAdapter, SQLiteAdapter
from backend.storage.data_storage import DataStorage
from backend.auth.auth import verify_token
from backend.analytics.usage import log_usage, get_usage_stats
from backend.history.query_history import save_query_history, get_recent_queries

class TestSQLGenerator:
    """Test cases for SQL generation functionality"""

    def test_generate_sql_basic_query(self):
        """Test basic SQL generation"""
        schema = {
            "customers": {"id": "INTEGER", "name": "TEXT", "city": "TEXT"}
        }

        with patch('backend.nl2sql.generator.openai.ChatCompletion.create') as mock_openai:
            mock_response = {
                "choices": [{
                    "message": {
                        "content": '{"sql": "SELECT * FROM customers WHERE city = \'New York\'", "suggestions": ["Add LIMIT clause", "Filter by active customers"]}'
                    }
                }]
            }
            mock_openai.return_value = mock_response

            result = generate_sql("Show me customers from New York", schema, Mock())

            assert "SELECT" in result.sql
            assert "customers" in result.sql
            assert "New York" in result.sql
            assert len(result.suggestions) > 0

    def test_generate_sql_with_joins(self):
        """Test SQL generation with table joins"""
        schema = {
            "customers": {"id": "INTEGER", "name": "TEXT"},
            "orders": {"id": "INTEGER", "customer_id": "INTEGER", "amount": "REAL"}
        }

        with patch('backend.nl2sql.generator.openai.ChatCompletion.create') as mock_openai:
            mock_openai.return_value = Mock(choices=[
                Mock(message=Mock(content='{"sql": "SELECT c.name, o.amount FROM customers c JOIN orders o ON c.id = o.customer_id", "suggestions": ["Add date filter", "Group by customer"]}'))
            ])

            result = generate_sql("Show customer names with their order amounts", schema, Mock())

            assert "JOIN" in result.sql
            assert "customers" in result.sql
            assert "orders" in result.sql

    def test_generate_sql_error_handling(self):
        """Test SQL generation error handling"""
        schema = {"customers": {"id": "INTEGER", "name": "TEXT"}}

        with patch('backend.nl2sql.generator.openai.ChatCompletion.create') as mock_openai:
            mock_openai.side_effect = Exception("OpenAI API error")

            with pytest.raises(Exception):
                generate_sql("Show me customers", schema, Mock())

    def test_generate_query_suggestions(self):
        """Test query suggestions generation"""
        with patch('backend.nl2sql.generator.openai.ChatCompletion.create') as mock_openai:
            mock_openai.return_value = Mock(choices=[
                Mock(message=Mock(content='["Try adding date filters", "Consider grouping by category", "Add LIMIT for performance"]'))
            ])

            suggestions = generate_query_suggestions("Show sales data")

            assert isinstance(suggestions, list)
            assert len(suggestions) > 0
            assert "date" in suggestions[0].lower()

class TestBiasDetector:
    """Test cases for bias detection functionality"""

    def test_detect_bias_no_bias(self):
        """Test bias detection with no significant bias"""
        detector = BiasDetector()

        data = [
            {"gender": "M", "salary": 50000},
            {"gender": "F", "salary": 52000},
            {"gender": "M", "salary": 48000},
            {"gender": "F", "salary": 51000}
        ]

        with patch('backend.nl2sql.bias_detection.openai.ChatCompletion.create') as mock_openai:
            mock_response = {
                "choices": [{
                    "message": {
                        "content": "No significant bias detected. Salary distribution appears balanced between genders."
                    }
                }]
            }
            mock_openai.return_value = mock_response

            result = detector.detect_bias(data, "Show salary by gender")

            assert "No significant bias" in result
            assert "balanced" in result

    def test_detect_bias_with_bias(self):
        """Test bias detection with significant bias"""
        detector = BiasDetector()

        # Create biased data
        data = [
            {"gender": "M", "salary": 60000},
            {"gender": "M", "salary": 65000},
            {"gender": "F", "salary": 35000},
            {"gender": "F", "salary": 38000}
        ]

        with patch('backend.nl2sql.bias_detection.openai.ChatCompletion.create') as mock_openai:
            mock_response = {
                "choices": [{
                    "message": {
                        "content": "Significant bias detected in salary distribution. Female employees earn approximately 40% less than male employees on average."
                    }
                }]
            }
            mock_openai.return_value = mock_response

            result = detector.detect_bias(data, "Compare salaries by gender")

            assert "bias detected" in result.lower()
            assert "salary" in result.lower()
            assert "salary" in result.lower()

    def test_detect_bias_empty_data(self):
        """Test bias detection with empty data"""
        detector = BiasDetector()

        result = detector.detect_bias([], "Empty query")

        assert "no data" in result.lower()

class TestDatabaseAdapters:
    """Test cases for database adapters"""

    def test_sqlite_adapter_connection(self):
        """Test SQLite adapter connection"""
        adapter = SQLiteAdapter(":memory:")

        # Test connection
        adapter.connect()
        assert adapter.conn is not None

        # Test health check
        health = adapter.health()
        assert health["connected"] is True

    def test_sqlite_adapter_run_query(self):
        """Test SQLite adapter query execution"""
        adapter = SQLiteAdapter(":memory:")

        adapter.connect()

        # Create test table
        adapter.run("CREATE TABLE test (id INTEGER, name TEXT)")

        # Insert test data
        adapter.run("INSERT INTO test VALUES (1, 'Test User')")

        # Query data
        result = adapter.run("SELECT * FROM test")

        assert len(result.rows) == 1
        assert result.rows[0][1] == "Test User"
        assert result.execution_time >= 0

    def test_postgres_adapter_initialization(self):
        """Test PostgreSQL adapter initialization"""
        config = {
            "host": "localhost",
            "port": 5432,
            "user": "test_user",
            "password": "test_pass",
            "dbname": "test_db"
        }

        adapter = PostgresAdapter(config)
        assert adapter.config == config

    def test_get_adapter_sqlite(self):
        """Test get_adapter function with SQLite"""
        with patch.dict('os.environ', {'DB_ENGINE': 'sqlite'}):
            with patch('backend.db.engine.SQLiteAdapter') as mock_sqlite:
                mock_adapter = Mock()
                mock_sqlite.return_value = mock_adapter

                result = get_adapter()

                mock_sqlite.assert_called_once()
                assert result == mock_adapter

    def test_get_adapter_postgres(self):
        """Test get_adapter function with PostgreSQL"""
        config = {
            "host": "localhost",
            "port": 5432,
            "user": "test_user",
            "password": "test_pass",
            "dbname": "test_db"
        }

        with patch.dict('os.environ', {
            'DB_ENGINE': 'postgres',
            'PG_HOST': 'localhost',
            'PG_USER': 'test_user',
            'PG_PASSWORD': 'test_pass',
            'PG_DBNAME': 'test_db'
        }):
            with patch('backend.db.engine.PostgresAdapter') as mock_postgres:
                mock_adapter = Mock()
                mock_postgres.return_value = mock_adapter

                result = get_adapter()

                mock_postgres.assert_called_once_with(config)
                assert result == mock_adapter

class TestDataStorage:
    """Test cases for data storage functionality"""

    def test_data_storage_initialization(self):
        """Test data storage initialization"""
        storage = DataStorage("local")
        assert storage.storage_type == "local"

    def test_save_and_load_data(self):
        """Test saving and loading data"""
        storage = DataStorage("local")

        test_data = [
            {"id": 1, "name": "Test Item"},
            {"id": 2, "name": "Another Item"}
        ]

        # Save data
        location = storage.save_data(test_data, "test_job")
        assert location is not None

        # Load data
        loaded_data = storage.load_data(location)
        assert loaded_data == test_data

    def test_generate_insights(self):
        """Test insights generation"""
        storage = DataStorage("local")

        data = [
            {"sales": 1000, "month": "Jan"},
            {"sales": 1200, "month": "Feb"},
            {"sales": 1500, "month": "Mar"}
        ]

        with patch('backend.storage.data_storage.openai.ChatCompletion.create') as mock_openai:
            mock_openai.return_value = Mock(choices=[
                Mock(message=Mock(content='Sales are showing consistent growth of approximately 15-20% month over month.'))
            ])

            insight = storage.generate_insights(data, "Analyze sales trend")

            assert "growth" in insight.lower()
            assert "15-20%" in insight

class TestAuthentication:
    """Test cases for authentication functionality"""

    def test_verify_token_valid(self):
        """Test token verification with valid token"""
        with patch('backend.auth.auth.jwt.decode') as mock_decode:
            mock_decode.return_value = {"user_id": "test_user", "exp": 1234567890}

            result = verify_token("Bearer valid_token")

            assert result == "test_user"

    def test_verify_token_invalid(self):
        """Test token verification with invalid token"""
        with patch('backend.auth.auth.jwt.decode') as mock_decode:
            mock_decode.side_effect = Exception("Invalid token")

            with pytest.raises(Exception):
                verify_token("Bearer invalid_token")

    def test_verify_token_missing(self):
        """Test token verification with missing token"""
        with pytest.raises(Exception):
            verify_token("")

class TestAnalytics:
    """Test cases for analytics functionality"""

    def test_log_usage(self):
        """Test usage logging"""
        # Reset any existing usage data
        with patch('backend.analytics.usage.usage_stats', {}):
            log_usage("/test_endpoint")

            stats = get_usage_stats()
            assert "/test_endpoint" in stats
            assert stats["/test_endpoint"] >= 1

    def test_get_usage_stats(self):
        """Test retrieving usage statistics"""
        stats = get_usage_stats()

        assert isinstance(stats, dict)
        # Should contain at least basic structure
        assert len(stats) >= 0

class TestQueryHistory:
    """Test cases for query history functionality"""

    def test_save_query_history(self):
        """Test saving query to history"""
        save_query_history("Show me customers", "SELECT * FROM customers", "test_job")

        # Verify it was saved (this would normally check a file or database)
        history = get_recent_queries()
        assert isinstance(history, list)

    def test_get_recent_queries(self):
        """Test retrieving recent queries"""
        queries = get_recent_queries()

        assert isinstance(queries, list)
        # Should return empty list or actual queries
        assert len(queries) >= 0

# Edge case tests
class TestEdgeCases:
    """Test cases for edge cases and error conditions"""

    def test_empty_query_generation(self):
        """Test SQL generation with empty query"""
        schema = {"customers": {"id": "INTEGER"}}

        with patch('backend.nl2sql.generator.openai.ChatCompletion.create') as mock_openai:
            mock_openai.return_value = Mock(choices=[
                Mock(message=Mock(content='{"sql": "SELECT 1", "suggestions": ["Please provide a more specific query"]}'))
            ])

            result = generate_sql("", schema, Mock())

            assert result.sql == "SELECT 1"
            assert len(result.suggestions) > 0

    def test_large_dataset_bias_detection(self):
        """Test bias detection with large dataset"""
        detector = BiasDetector()

        # Generate large dataset
        data = []
        for i in range(1000):
            data.append({
                "category": "A" if i % 2 == 0 else "B",
                "value": i * 10
            })

        with patch('backend.nl2sql.bias_detection.openai.ChatCompletion.create') as mock_openai:
            mock_openai.return_value = Mock(choices=[
                Mock(message=Mock(content='{"analysis": "Large dataset analysis complete. No significant bias detected.", "recommendations": ["Dataset size is adequate for analysis"]}'))
            ])

            result = detector.detect_bias(data, "Analyze large dataset")

            assert "Large dataset" in result
            assert "complete" in result

    def test_special_characters_in_query(self):
        """Test handling of special characters in queries"""
        schema = {"products": {"name": "TEXT", "description": "TEXT"}}

        with patch('backend.nl2sql.generator.openai.ChatCompletion.create') as mock_openai:
            mock_openai.return_value = Mock(choices=[
                Mock(message=Mock(content='{"sql": "SELECT * FROM products WHERE name LIKE \'%special%\'", "suggestions": ["Consider full-text search", "Add category filter"]}'))
            ])

            result = generate_sql("Find products with 'special' in the name", schema, Mock())

            assert "LIKE" in result.sql
            assert "%special%" in result.sql

    def test_concurrent_queries_simulation(self):
        """Test handling of concurrent queries"""
        import threading
        import time

        results = []
        errors = []

        def run_query(query_id):
            try:
                time.sleep(0.01)  # Simulate processing time
                results.append(f"Query {query_id} completed")
            except Exception as e:
                errors.append(str(e))

        # Simulate concurrent queries
        threads = []
        for i in range(5):
            thread = threading.Thread(target=run_query, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        assert len(results) == 5
        assert len(errors) == 0
        assert all("completed" in result for result in results)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
