"""
Custom middleware for NL2Q Analyst V2
"""
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from src.core.config import settings

logger = structlog.get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        if settings.https_only:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log incoming requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log incoming request
        start_time = time.time()
        
        logger.info(
            "Incoming request",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            user_agent=request.headers.get("user-agent"),
            client_ip=request.client.host if request.client else None
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        
        logger.info(
            "Request completed",
            request_id=request_id,
            status_code=response.status_code,
            process_time=f"{process_time:.3f}s"
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""
    
    def __init__(self, app, calls: int = None, period: int = 60):
        super().__init__(app)
        self.calls = calls or settings.rate_limit_requests_per_minute
        self.period = period
        self.clients = {}
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/health", "/", f"{settings.api_v2_prefix}/docs"]:
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean up old entries
        cutoff_time = current_time - self.period
        self.clients = {
            ip: timestamps for ip, timestamps in self.clients.items()
            if any(t > cutoff_time for t in timestamps)
        }
        
        # Check rate limit
        if client_ip in self.clients:
            self.clients[client_ip] = [
                t for t in self.clients[client_ip] if t > cutoff_time
            ]
            
            if len(self.clients[client_ip]) >= self.calls:
                logger.warning(
                    "Rate limit exceeded",
                    client_ip=client_ip,
                    requests=len(self.clients[client_ip])
                )
                
                return Response(
                    content='{"error": true, "message": "Rate limit exceeded"}',
                    status_code=429,
                    headers={"Content-Type": "application/json"}
                )
        else:
            self.clients[client_ip] = []
        
        # Record this request
        self.clients[client_ip].append(current_time)
        
        return await call_next(request)


class TenantMiddleware(BaseHTTPMiddleware):
    """Handle multi-tenant context."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Extract tenant ID from header or use default
        tenant_id = request.headers.get("X-Tenant-ID", settings.default_tenant_id)
        request.state.tenant_id = tenant_id
        
        logger.debug("Tenant context set", tenant_id=tenant_id)
        
        response = await call_next(request)
        response.headers["X-Tenant-ID"] = tenant_id
        
        return response