"""
Custom exceptions for NL2Q Analyst V2
"""
from typing import Any, Dict, Optional


class NL2QException(Exception):
    """Base exception class for NL2Q Analyst."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "NL2Q_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class DatabaseConnectionError(NL2QException):
    """Raised when database connection fails."""
    
    def __init__(self, message: str = "Database connection failed", **kwargs):
        super().__init__(
            message=message,
            error_code="DATABASE_CONNECTION_ERROR",
            status_code=503,
            **kwargs
        )


class LLMProviderError(NL2QException):
    """Raised when LLM provider fails."""
    
    def __init__(self, message: str = "LLM provider error", provider: str = None, **kwargs):
        details = kwargs.get("details", {})
        if provider:
            details["provider"] = provider
            
        super().__init__(
            message=message,
            error_code="LLM_PROVIDER_ERROR",
            status_code=502,
            details=details,
            **kwargs
        )


class QueryExecutionError(NL2QException):
    """Raised when query execution fails."""
    
    def __init__(self, message: str = "Query execution failed", **kwargs):
        super().__init__(
            message=message,
            error_code="QUERY_EXECUTION_ERROR",
            status_code=400,
            **kwargs
        )


class AuthenticationError(NL2QException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
            **kwargs
        )


class AuthorizationError(NL2QException):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            **kwargs
        )


class ResourceNotFoundError(NL2QException):
    """Raised when a resource is not found."""
    
    def __init__(self, message: str = "Resource not found", **kwargs):
        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            status_code=404,
            **kwargs
        )


class ValidationError(NL2QException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str = "Validation failed", **kwargs):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=422,
            **kwargs
        )


class RateLimitError(NL2QException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", **kwargs):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            **kwargs
        )


class TenantError(NL2QException):
    """Raised when tenant-related errors occur."""
    
    def __init__(self, message: str = "Tenant error", **kwargs):
        super().__init__(
            message=message,
            error_code="TENANT_ERROR",
            status_code=400,
            **kwargs
        )