"""
Main FastAPI application for NL2Q Analyst V2
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from src.core.config import settings
from src.api.routes import api_router
from src.core.exceptions import NL2QException
from src.core.middleware import (
    SecurityHeadersMiddleware,
    RequestLoggingMiddleware,
    RateLimitMiddleware
)
from src.services.cache import cache_service
from src.services.database import database_service
from src.services.llm import llm_service

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("🚀 Starting NL2Q Analyst V2", version=settings.version)
    
    # Initialize services
    await cache_service.initialize()
    await database_service.initialize()
    await llm_service.initialize()
    
    logger.info("✅ All services initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down NL2Q Analyst V2")
    
    # Cleanup services
    await cache_service.cleanup()
    await database_service.cleanup()
    await llm_service.cleanup()
    
    logger.info("✅ Shutdown completed")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Next-Generation Natural Language to Query Platform",
    docs_url=f"{settings.api_v2_prefix}/docs",
    redoc_url=f"{settings.api_v2_prefix}/redoc",
    openapi_url=f"{settings.api_v2_prefix}/openapi.json",
    lifespan=lifespan,
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)

# Add custom middleware
if settings.security_headers_enabled:
    app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)

# Include API routes
app.include_router(api_router, prefix=settings.api_v2_prefix)


@app.exception_handler(NL2QException)
async def nl2q_exception_handler(request: Request, exc: NL2QException):
    """Handle custom NL2Q exceptions."""
    logger.error("NL2Q Exception", error=str(exc), error_code=exc.error_code)
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error("Unexpected error", error=str(exc), error_type=type(exc).__name__)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "details": None if not settings.debug else str(exc)
        }
    )


@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "status": "running",
        "docs": f"{settings.api_v2_prefix}/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "version": settings.version,
        "services": {}
    }
    
    # Check cache service
    try:
        await cache_service.ping()
        health_status["services"]["cache"] = "healthy"
    except Exception as e:
        health_status["services"]["cache"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check database service
    try:
        await database_service.ping()
        health_status["services"]["database"] = "healthy"
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check LLM service
    try:
        await llm_service.ping()
        health_status["services"]["llm"] = "healthy"
    except Exception as e:
        health_status["services"]["llm"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )