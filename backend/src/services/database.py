"""
Database Service for NL2Q Analyst V2

Multi-database support with connection pooling and schema intelligence.
"""
import hashlib
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from enum import Enum
import structlog

from src.core.config import settings
from src.core.exceptions import DatabaseConnectionError
from src.services.cache import cache_service

logger = structlog.get_logger(__name__)


class DatabaseType(Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    REDSHIFT = "redshift"
    MONGODB = "mongodb"
    SQLITE = "sqlite"


class BaseDatabaseAdapter(ABC):
    """Abstract base class for database adapters."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the database."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close database connection."""
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a query and return results."""
        pass
    
    @abstractmethod
    async def get_schema(self) -> Dict[str, Any]:
        """Get database schema information."""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test database connection."""
        pass


class PostgreSQLAdapter(BaseDatabaseAdapter):
    """PostgreSQL database adapter."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = None
        self.async_session = None
    
    async def connect(self) -> bool:
        """Connect to PostgreSQL."""
        try:
            from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
            from sqlalchemy.orm import sessionmaker
            
            self.engine = create_async_engine(
                self.connection_string.replace("postgresql://", "postgresql+asyncpg://"),
                echo=settings.debug,
                pool_size=10,
                max_overflow=20
            )
            
            self.async_session = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
            
            # Test connection
            await self.test_connection()
            logger.info("PostgreSQL connection established")
            return True
            
        except Exception as e:
            logger.error("Failed to connect to PostgreSQL", error=str(e))
            raise DatabaseConnectionError(f"PostgreSQL connection failed: {str(e)}")
    
    async def disconnect(self):
        """Disconnect from PostgreSQL."""
        if self.engine:
            await self.engine.dispose()
            logger.info("PostgreSQL connection closed")
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute PostgreSQL query."""
        try:
            from sqlalchemy import text
            
            async with self.async_session() as session:
                result = await session.execute(text(query), params or {})
                
                if result.returns_rows:
                    rows = result.fetchall()
                    columns = result.keys()
                    
                    return {
                        "columns": list(columns),
                        "rows": [dict(zip(columns, row)) for row in rows],
                        "row_count": len(rows)
                    }
                else:
                    return {
                        "columns": [],
                        "rows": [],
                        "row_count": result.rowcount,
                        "affected_rows": result.rowcount
                    }
                    
        except Exception as e:
            logger.error("PostgreSQL query execution failed", query=query, error=str(e))
            raise DatabaseConnectionError(f"Query execution failed: {str(e)}")
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get PostgreSQL schema."""
        schema_query = """
        SELECT 
            table_schema,
            table_name,
            column_name,
            data_type,
            is_nullable,
            column_default,
            ordinal_position
        FROM information_schema.columns
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
        ORDER BY table_schema, table_name, ordinal_position
        """
        
        result = await self.execute_query(schema_query)
        
        # Organize schema by table
        schema = {}
        for row in result["rows"]:
            table_key = f"{row['table_schema']}.{row['table_name']}"
            if table_key not in schema:
                schema[table_key] = {
                    "schema": row["table_schema"],
                    "table": row["table_name"],
                    "columns": []
                }
            
            schema[table_key]["columns"].append({
                "name": row["column_name"],
                "type": row["data_type"],
                "nullable": row["is_nullable"] == "YES",
                "default": row["column_default"],
                "position": row["ordinal_position"]
            })
        
        return {
            "database_type": "postgresql",
            "tables": schema,
            "table_count": len(schema)
        }
    
    async def test_connection(self) -> bool:
        """Test PostgreSQL connection."""
        try:
            result = await self.execute_query("SELECT 1 as test")
            return result["rows"][0]["test"] == 1
        except Exception:
            return False


class SnowflakeAdapter(BaseDatabaseAdapter):
    """Snowflake database adapter."""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.connection = None
    
    async def connect(self) -> bool:
        """Connect to Snowflake."""
        try:
            import snowflake.connector
            
            self.connection = snowflake.connector.connect(
                account=self.config["account"],
                user=self.config["user"],
                password=self.config["password"],
                warehouse=self.config.get("warehouse", "COMPUTE_WH"),
                database=self.config.get("database"),
                schema=self.config.get("schema", "PUBLIC")
            )
            
            logger.info("Snowflake connection established")
            return True
            
        except Exception as e:
            logger.error("Failed to connect to Snowflake", error=str(e))
            raise DatabaseConnectionError(f"Snowflake connection failed: {str(e)}")
    
    async def disconnect(self):
        """Disconnect from Snowflake."""
        if self.connection:
            self.connection.close()
            logger.info("Snowflake connection closed")
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute Snowflake query."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params or {})
            
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                return {
                    "columns": columns,
                    "rows": [dict(zip(columns, row)) for row in rows],
                    "row_count": len(rows)
                }
            else:
                return {
                    "columns": [],
                    "rows": [],
                    "row_count": cursor.rowcount,
                    "affected_rows": cursor.rowcount
                }
                
        except Exception as e:
            logger.error("Snowflake query execution failed", query=query, error=str(e))
            raise DatabaseConnectionError(f"Query execution failed: {str(e)}")
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get Snowflake schema."""
        schema_query = """
        SELECT 
            table_schema,
            table_name,
            column_name,
            data_type,
            is_nullable,
            column_default,
            ordinal_position
        FROM information_schema.columns
        WHERE table_schema != 'INFORMATION_SCHEMA'
        ORDER BY table_schema, table_name, ordinal_position
        """
        
        result = await self.execute_query(schema_query)
        
        # Organize schema by table
        schema = {}
        for row in result["rows"]:
            table_key = f"{row['TABLE_SCHEMA']}.{row['TABLE_NAME']}"
            if table_key not in schema:
                schema[table_key] = {
                    "schema": row["TABLE_SCHEMA"],
                    "table": row["TABLE_NAME"],
                    "columns": []
                }
            
            schema[table_key]["columns"].append({
                "name": row["COLUMN_NAME"],
                "type": row["DATA_TYPE"],
                "nullable": row["IS_NULLABLE"] == "YES",
                "default": row["COLUMN_DEFAULT"],
                "position": row["ORDINAL_POSITION"]
            })
        
        return {
            "database_type": "snowflake",
            "tables": schema,
            "table_count": len(schema)
        }
    
    async def test_connection(self) -> bool:
        """Test Snowflake connection."""
        try:
            result = await self.execute_query("SELECT 1 as test")
            return result["rows"][0]["test"] == 1
        except Exception:
            return False


class DatabaseService:
    """Multi-database service with connection management."""
    
    def __init__(self):
        self.adapters: Dict[str, BaseDatabaseAdapter] = {}
        self.default_adapter_id = "default"
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connections."""
        if self._initialized:
            return
        
        logger.info("Initializing database service")
        
        # Initialize default PostgreSQL/SQLite connection
        if settings.database_url:
            try:
                if settings.database_url.startswith("postgresql"):
                    adapter = PostgreSQLAdapter(settings.database_url)
                else:
                    # For SQLite and other databases, we'd add more adapters
                    logger.warning("Only PostgreSQL fully implemented in this demo")
                    adapter = None
                
                if adapter:
                    await adapter.connect()
                    self.adapters[self.default_adapter_id] = adapter
                    logger.info("Default database connection established")
                    
            except Exception as e:
                logger.error("Failed to initialize default database", error=str(e))
        
        # Initialize Snowflake if configured
        if all([
            settings.snowflake_account,
            settings.snowflake_user,
            settings.snowflake_password
        ]):
            try:
                snowflake_config = {
                    "account": settings.snowflake_account,
                    "user": settings.snowflake_user,
                    "password": settings.snowflake_password,
                    "warehouse": settings.snowflake_warehouse,
                    "database": settings.snowflake_database,
                    "schema": settings.snowflake_schema
                }
                
                adapter = SnowflakeAdapter(snowflake_config)
                await adapter.connect()
                self.adapters["snowflake"] = adapter
                logger.info("Snowflake connection established")
                
            except Exception as e:
                logger.warning("Failed to initialize Snowflake", error=str(e))
        
        self._initialized = True
        logger.info("Database service initialized", adapters=list(self.adapters.keys()))
    
    async def execute_query(
        self,
        query: str,
        database_id: str = None,
        params: Optional[Dict] = None,
        tenant_id: str = "default"
    ) -> Dict[str, Any]:
        """Execute query on specified database."""
        database_id = database_id or self.default_adapter_id
        
        if database_id not in self.adapters:
            raise DatabaseConnectionError(f"Database adapter '{database_id}' not found")
        
        adapter = self.adapters[database_id]
        
        try:
            # Check cache first
            cache_key = self._generate_query_cache_key(query, database_id, tenant_id)
            cached_result = await cache_service.get(cache_key)
            
            if cached_result:
                logger.info("Query result served from cache", database_id=database_id)
                cached_result["from_cache"] = True
                return cached_result
            
            # Execute query
            result = await adapter.execute_query(query, params)
            result["database_id"] = database_id
            result["from_cache"] = False
            
            # Cache result if it's a SELECT query
            if query.strip().upper().startswith("SELECT"):
                await cache_service.set_query_cache(
                    query, self._generate_schema_hash(database_id), result, tenant_id
                )
            
            logger.info("Query executed successfully", database_id=database_id, rows=result.get("row_count", 0))
            return result
            
        except Exception as e:
            logger.error("Query execution failed", database_id=database_id, query=query, error=str(e))
            raise
    
    async def get_schema(
        self,
        database_id: str = None,
        tenant_id: str = "default"
    ) -> Dict[str, Any]:
        """Get database schema with caching."""
        database_id = database_id or self.default_adapter_id
        
        if database_id not in self.adapters:
            raise DatabaseConnectionError(f"Database adapter '{database_id}' not found")
        
        # Check cache first
        cached_schema = await cache_service.get_schema_cache(database_id, tenant_id)
        if cached_schema:
            logger.info("Schema served from cache", database_id=database_id)
            return cached_schema
        
        adapter = self.adapters[database_id]
        
        try:
            schema = await adapter.get_schema()
            schema["database_id"] = database_id
            
            # Cache schema
            await cache_service.set_schema_cache(database_id, schema, tenant_id)
            
            logger.info("Schema retrieved successfully", database_id=database_id, tables=schema.get("table_count", 0))
            return schema
            
        except Exception as e:
            logger.error("Schema retrieval failed", database_id=database_id, error=str(e))
            raise
    
    async def ping(self) -> Dict[str, bool]:
        """Test all database connections."""
        status = {}
        
        for adapter_id, adapter in self.adapters.items():
            try:
                status[adapter_id] = await adapter.test_connection()
            except Exception:
                status[adapter_id] = False
        
        return status
    
    async def cleanup(self):
        """Cleanup all database connections."""
        logger.info("Cleaning up database service")
        
        for adapter_id, adapter in self.adapters.items():
            try:
                await adapter.disconnect()
            except Exception as e:
                logger.warning("Failed to disconnect adapter", adapter_id=adapter_id, error=str(e))
        
        self.adapters.clear()
        self._initialized = False
    
    def get_available_databases(self) -> List[str]:
        """Get list of available database connections."""
        return list(self.adapters.keys())
    
    def _generate_query_cache_key(self, query: str, database_id: str, tenant_id: str) -> str:
        """Generate cache key for query."""
        content = f"{tenant_id}:{database_id}:{query}"
        return f"query:{hashlib.md5(content.encode()).hexdigest()}"
    
    def _generate_schema_hash(self, database_id: str) -> str:
        """Generate schema hash for cache keys."""
        return f"schema:{database_id}"


# Global database service instance
database_service = DatabaseService()