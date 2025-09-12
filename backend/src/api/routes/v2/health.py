"""
Health check endpoints for NL2Q Analyst V2
"""
from fastapi import APIRouter
from typing import Dict, Any
import structlog

from src.services.llm import llm_service
from src.services.database import database_service
from src.services.cache import cache_service
from src.core.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check endpoint.
    
    Returns the health status of all major components:
    - Application status
    - Database connections
    - Cache service
    - LLM providers
    """
    health_status = {
        "status": "healthy",
        "version": settings.version,
        "environment": settings.environment,
        "timestamp": None,
        "services": {}
    }
    
    # Import here to avoid circular imports
    from datetime import datetime
    health_status["timestamp"] = datetime.utcnow().isoformat()
    
    # Check cache service
    try:
        cache_ping = await cache_service.ping()
        cache_stats = await cache_service.get_cache_stats()
        
        health_status["services"]["cache"] = {
            "status": "healthy" if cache_ping else "unhealthy",
            "stats": cache_stats
        }
        
        if not cache_ping:
            health_status["status"] = "degraded"
            
    except Exception as e:
        health_status["services"]["cache"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check database service
    try:
        db_status = await database_service.ping()
        
        health_status["services"]["database"] = {
            "status": "healthy" if all(db_status.values()) else "degraded",
            "connections": db_status,
            "available_databases": database_service.get_available_databases()
        }
        
        if not all(db_status.values()):
            health_status["status"] = "degraded"
            
    except Exception as e:
        health_status["services"]["database"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check LLM service
    try:
        llm_status = await llm_service.ping()
        
        health_status["services"]["llm"] = {
            "status": "healthy" if any(llm_status.values()) else "unhealthy",
            "providers": llm_status,
            "available_providers": llm_service.get_available_providers(),
            "default_provider": settings.default_llm_provider
        }
        
        if not any(llm_status.values()):
            health_status["status"] = "degraded"
            
    except Exception as e:
        health_status["services"]["llm"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Feature flags status
    health_status["features"] = {
        "ml_suggestions": settings.enable_ml_suggestions,
        "real_time_collab": settings.enable_real_time_collab,
        "multi_tenant": settings.enable_multi_tenant,
        "query_caching": settings.enable_query_caching,
        "streaming_responses": settings.enable_streaming_responses,
        "advanced_analytics": settings.enable_advanced_analytics
    }
    
    logger.info("Health check completed", status=health_status["status"])
    
    return health_status


@router.get("/live")
async def liveness_probe():
    """
    Simple liveness probe for Kubernetes/container orchestration.
    Returns 200 if the application is running.
    """
    return {"status": "alive", "service": settings.app_name}


@router.get("/ready")
async def readiness_probe():
    """
    Readiness probe to check if the service is ready to handle requests.
    Checks critical dependencies.
    """
    ready = True
    checks = {}
    
    # Check if at least one database is available
    try:
        db_status = await database_service.ping()
        db_ready = any(db_status.values()) if db_status else False
        checks["database"] = db_ready
        if not db_ready:
            ready = False
    except Exception:
        checks["database"] = False
        ready = False
    
    # Check if at least one LLM provider is available
    try:
        llm_status = await llm_service.ping()
        llm_ready = any(llm_status.values()) if llm_status else False
        checks["llm"] = llm_ready
        if not llm_ready:
            ready = False
    except Exception:
        checks["llm"] = False
        ready = False
    
    status_code = 200 if ready else 503
    
    return {
        "status": "ready" if ready else "not_ready",
        "checks": checks
    }


@router.get("/metrics")
async def get_metrics():
    """
    Get application metrics for monitoring.
    """
    metrics = {
        "application": {
            "name": settings.app_name,
            "version": settings.version,
            "environment": settings.environment
        },
        "system": {},
        "services": {}
    }
    
    # Get cache metrics
    try:
        cache_stats = await cache_service.get_cache_stats()
        metrics["services"]["cache"] = cache_stats
    except Exception as e:
        metrics["services"]["cache"] = {"error": str(e)}
    
    # Get database metrics
    try:
        db_status = await database_service.ping()
        metrics["services"]["database"] = {
            "connections": db_status,
            "available_count": len([s for s in db_status.values() if s])
        }
    except Exception as e:
        metrics["services"]["database"] = {"error": str(e)}
    
    # Get LLM metrics
    try:
        llm_status = await llm_service.ping()
        metrics["services"]["llm"] = {
            "providers": llm_status,
            "available_count": len([s for s in llm_status.values() if s])
        }
    except Exception as e:
        metrics["services"]["llm"] = {"error": str(e)}
    
    return metrics