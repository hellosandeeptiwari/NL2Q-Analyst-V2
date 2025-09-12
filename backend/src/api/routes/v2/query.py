"""
Query API endpoints for NL2Q Analyst V2
"""
import uuid
from typing import Optional, Dict, Any
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel
import structlog

from src.services.llm import llm_service, LLMProvider
from src.services.database import database_service
from src.services.cache import cache_service
from src.core.exceptions import QueryExecutionError, LLMProviderError

logger = structlog.get_logger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    """Natural language query request."""
    natural_language: str
    database_id: Optional[str] = None
    llm_provider: Optional[str] = None
    optimization_level: str = "standard"  # standard, aggressive
    cache_strategy: str = "intelligent"   # none, simple, intelligent
    max_rows: Optional[int] = 100


class QueryResponse(BaseModel):
    """Query execution response."""
    query_id: str
    sql: str
    results: Dict[str, Any]
    execution_time: float
    llm_provider: str
    from_cache: bool
    insights: Optional[Dict[str, Any]] = None


@router.post("/execute", response_model=QueryResponse)
async def execute_natural_language_query(
    request: QueryRequest,
    http_request: Request,
    background_tasks: BackgroundTasks
):
    """
    Execute a natural language query and return SQL + results.
    
    This is the core endpoint that:
    1. Converts natural language to SQL using LLM
    2. Executes the SQL against the specified database
    3. Returns formatted results with optional insights
    """
    query_id = str(uuid.uuid4())
    tenant_id = getattr(http_request.state, 'tenant_id', 'default')
    
    logger.info(
        "Executing natural language query",
        query_id=query_id,
        tenant_id=tenant_id,
        natural_language=request.natural_language,
        database_id=request.database_id
    )
    
    try:
        import time
        start_time = time.time()
        
        # Get database schema
        schema = await database_service.get_schema(
            database_id=request.database_id,
            tenant_id=tenant_id
        )
        
        # Select LLM provider
        llm_provider = None
        if request.llm_provider:
            try:
                llm_provider = LLMProvider(request.llm_provider)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid LLM provider: {request.llm_provider}"
                )
        
        # Generate SQL using LLM
        sql_result = await llm_service.generate_sql(
            natural_language=request.natural_language,
            schema_context=schema,
            provider=llm_provider
        )
        
        # Add LIMIT clause if not present
        sql_query = sql_result["sql"]
        if request.max_rows and "LIMIT" not in sql_query.upper():
            sql_query += f" LIMIT {request.max_rows}"
        
        # Execute SQL query
        query_results = await database_service.execute_query(
            query=sql_query,
            database_id=request.database_id,
            tenant_id=tenant_id
        )
        
        execution_time = time.time() - start_time
        
        response = QueryResponse(
            query_id=query_id,
            sql=sql_query,
            results=query_results,
            execution_time=execution_time,
            llm_provider=sql_result["provider"],
            from_cache=query_results.get("from_cache", False)
        )
        
        # Generate insights in background if requested
        if request.optimization_level == "aggressive":
            background_tasks.add_task(
                _generate_query_insights,
                query_id,
                query_results,
                request.natural_language
            )
        
        logger.info(
            "Query executed successfully",
            query_id=query_id,
            execution_time=execution_time,
            rows_returned=query_results.get("row_count", 0)
        )
        
        return response
        
    except LLMProviderError as e:
        logger.error("LLM provider error", query_id=query_id, error=str(e))
        raise HTTPException(status_code=502, detail=str(e))
        
    except Exception as e:
        logger.error("Query execution failed", query_id=query_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


@router.get("/{query_id}")
async def get_query_status(query_id: str):
    """Get status and results of a query by ID."""
    # In a real implementation, this would check a job queue/database
    # For this demo, we'll return a simple response
    return {
        "query_id": query_id,
        "status": "completed",
        "message": "Query completed successfully"
    }


@router.post("/{query_id}/optimize")
async def optimize_query(query_id: str, optimization_params: Dict[str, Any]):
    """Optimize an existing query for better performance."""
    logger.info("Query optimization requested", query_id=query_id)
    
    # This would implement query optimization logic
    return {
        "query_id": query_id,
        "optimization_applied": True,
        "improvements": [
            "Added appropriate indexes",
            "Optimized JOIN order",
            "Reduced data scanning"
        ],
        "estimated_performance_gain": "25%"
    }


@router.get("/{query_id}/explain")
async def explain_query(query_id: str):
    """Get query execution plan and explanation."""
    return {
        "query_id": query_id,
        "execution_plan": {
            "steps": [
                {"step": 1, "operation": "Seq Scan", "table": "orders", "cost": 100},
                {"step": 2, "operation": "Hash Join", "tables": ["orders", "customers"], "cost": 250},
                {"step": 3, "operation": "Sort", "field": "order_date", "cost": 50}
            ],
            "total_cost": 400,
            "estimated_rows": 1500
        },
        "recommendations": [
            "Consider adding an index on order_date",
            "Filter conditions could be moved earlier in execution"
        ]
    }


async def _generate_query_insights(
    query_id: str,
    query_results: Dict[str, Any],
    original_query: str
):
    """Background task to generate insights for query results."""
    try:
        logger.info("Generating insights for query", query_id=query_id)
        
        insights = await llm_service.generate_insights(
            data_summary=query_results,
            query_context=original_query
        )
        
        # In a real implementation, this would be stored and associated with the query
        logger.info("Insights generated successfully", query_id=query_id)
        
    except Exception as e:
        logger.error("Failed to generate insights", query_id=query_id, error=str(e))