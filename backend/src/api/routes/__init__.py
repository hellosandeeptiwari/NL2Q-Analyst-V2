"""
API Routes for NL2Q Analyst V2
"""
from fastapi import APIRouter

from .v2 import query, llm, analytics, database, health

# Create main API router
api_router = APIRouter()

# Include all route modules
api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(query.router, prefix="/query", tags=["Query"])
api_router.include_router(llm.router, prefix="/llm", tags=["LLM"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
api_router.include_router(database.router, prefix="/database", tags=["Database"])