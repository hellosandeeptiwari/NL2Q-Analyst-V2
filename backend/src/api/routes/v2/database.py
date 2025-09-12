"""
Database management endpoints for NL2Q Analyst V2
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from pydantic import BaseModel
import structlog

from src.services.database import database_service
from src.services.cache import cache_service

logger = structlog.get_logger(__name__)

router = APIRouter()


class DatabaseInfo(BaseModel):
    """Database connection information."""
    id: str
    type: str
    status: bool
    table_count: int


@router.get("/connections", response_model=List[DatabaseInfo])
async def get_database_connections():
    """
    Get list of available database connections and their status.
    """
    try:
        db_status = await database_service.ping()
        connections = []
        
        for db_id, status in db_status.items():
            # Get schema to determine table count
            try:
                schema = await database_service.get_schema(db_id)
                table_count = schema.get("table_count", 0)
                db_type = schema.get("database_type", "unknown")
            except Exception:
                table_count = 0
                db_type = "unknown"
            
            connections.append(DatabaseInfo(
                id=db_id,
                type=db_type,
                status=status,
                table_count=table_count
            ))
        
        return connections
        
    except Exception as e:
        logger.error("Failed to get database connections", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve database information")


@router.get("/{database_id}/schema")
async def get_database_schema(database_id: str):
    """
    Get detailed schema information for a specific database.
    """
    try:
        schema = await database_service.get_schema(database_id)
        return schema
        
    except Exception as e:
        logger.error("Failed to get database schema", database_id=database_id, error=str(e))
        raise HTTPException(status_code=404, detail=f"Schema not found for database: {database_id}")


@router.get("/{database_id}/tables")
async def get_database_tables(database_id: str):
    """
    Get list of tables in a specific database.
    """
    try:
        schema = await database_service.get_schema(database_id)
        tables = []
        
        for table_key, table_info in schema.get("tables", {}).items():
            tables.append({
                "name": table_info["table"],
                "schema": table_info.get("schema", "public"),
                "full_name": table_key,
                "column_count": len(table_info.get("columns", []))
            })
        
        return {
            "database_id": database_id,
            "tables": tables,
            "total_count": len(tables)
        }
        
    except Exception as e:
        logger.error("Failed to get database tables", database_id=database_id, error=str(e))
        raise HTTPException(status_code=404, detail=f"Tables not found for database: {database_id}")


@router.get("/{database_id}/tables/{table_name}")
async def get_table_details(database_id: str, table_name: str):
    """
    Get detailed information about a specific table.
    """
    try:
        schema = await database_service.get_schema(database_id)
        tables = schema.get("tables", {})
        
        # Find the table (handle both simple name and schema.table format)
        table_info = None
        for table_key, info in tables.items():
            if info["table"] == table_name or table_key == table_name:
                table_info = info
                break
        
        if not table_info:
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
        
        # Get sample data (first few rows)
        try:
            sample_query = f"SELECT * FROM {table_name} LIMIT 5"
            sample_data = await database_service.execute_query(
                query=sample_query,
                database_id=database_id
            )
        except Exception as e:
            logger.warning("Failed to get sample data", table=table_name, error=str(e))
            sample_data = {"rows": [], "columns": []}
        
        return {
            "database_id": database_id,
            "table_name": table_name,
            "schema": table_info.get("schema", "public"),
            "columns": table_info.get("columns", []),
            "sample_data": sample_data,
            "metadata": {
                "column_count": len(table_info.get("columns", [])),
                "estimated_rows": "unknown"  # Would need additional queries to determine
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get table details", table_name=table_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve table details: {str(e)}")


@router.post("/{database_id}/test")
async def test_database_connection(database_id: str):
    """
    Test connection to a specific database.
    """
    try:
        status = await database_service.ping()
        db_status = status.get(database_id, False)
        
        if db_status:
            # Try a simple query to verify full functionality
            test_query = "SELECT 1 as test_connection"
            result = await database_service.execute_query(
                query=test_query,
                database_id=database_id
            )
            
            return {
                "database_id": database_id,
                "status": "healthy",
                "connection": "successful",
                "test_query_result": result
            }
        else:
            return {
                "database_id": database_id,
                "status": "unhealthy",
                "connection": "failed",
                "error": "Connection test failed"
            }
            
    except Exception as e:
        logger.error("Database connection test failed", database_id=database_id, error=str(e))
        return {
            "database_id": database_id,
            "status": "error",
            "connection": "failed",
            "error": str(e)
        }


@router.delete("/{database_id}/cache")
async def clear_database_cache(database_id: str, tenant_id: str = "default"):
    """
    Clear cache for a specific database.
    """
    try:
        # Clear schema cache
        schema_cleared = await cache_service.delete(
            cache_service._generate_schema_cache_key(database_id, tenant_id)
        )
        
        # In a real implementation, we'd clear all query caches related to this database
        # For now, we'll just clear the tenant cache
        queries_cleared = await cache_service.invalidate_tenant_cache(tenant_id)
        
        logger.info("Database cache cleared", database_id=database_id, tenant_id=tenant_id)
        
        return {
            "database_id": database_id,
            "tenant_id": tenant_id,
            "schema_cache_cleared": schema_cleared,
            "query_caches_cleared": queries_cleared,
            "status": "success"
        }
        
    except Exception as e:
        logger.error("Failed to clear database cache", database_id=database_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/{database_id}/stats")
async def get_database_statistics(database_id: str):
    """
    Get statistics and usage information for a database.
    """
    try:
        # In a real implementation, this would track actual usage statistics
        return {
            "database_id": database_id,
            "statistics": {
                "total_queries": 1543,
                "successful_queries": 1465,
                "failed_queries": 78,
                "success_rate": 0.949,
                "avg_query_time": 2.3,
                "most_queried_tables": [
                    {"table": "orders", "query_count": 234},
                    {"table": "customers", "query_count": 189},
                    {"table": "products", "query_count": 156}
                ],
                "query_patterns": {
                    "aggregations": 45.2,
                    "filters": 32.1,
                    "joins": 18.7,
                    "others": 4.0
                }
            },
            "performance": {
                "cache_hit_rate": 0.73,
                "avg_response_time": 1.8,
                "peak_concurrent_queries": 8
            }
        }
        
    except Exception as e:
        logger.error("Failed to get database statistics", database_id=database_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")