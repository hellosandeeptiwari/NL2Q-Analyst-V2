"""
NL2Q Analyst V2 - Configuration Settings

Centralized configuration management for the application.
"""
from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "NL2Q-Analyst-V2"
    version: str = "2.0.0"
    debug: bool = False
    environment: str = "development"
    
    # API Configuration
    api_v2_prefix: str = "/api/v2"
    cors_origins: List[str] = ["http://localhost:3000"]
    max_query_execution_time: int = 300
    max_concurrent_queries: int = 10
    
    # AI/LLM Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    default_llm_provider: str = "openai"
    
    # Database Configuration
    database_url: str = "sqlite:///./nl2q_v2.db"
    mongodb_uri: Optional[str] = None
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl: int = 3600
    
    # Snowflake Configuration
    snowflake_account: Optional[str] = None
    snowflake_user: Optional[str] = None
    snowflake_password: Optional[str] = None
    snowflake_warehouse: str = "COMPUTE_WH"
    snowflake_database: Optional[str] = None
    snowflake_schema: str = "PUBLIC"
    
    # BigQuery Configuration
    bigquery_project_id: Optional[str] = None
    
    # Redshift Configuration
    redshift_host: Optional[str] = None
    redshift_port: int = 5439
    redshift_user: Optional[str] = None
    redshift_password: Optional[str] = None
    redshift_database: Optional[str] = None
    
    # Vector Store Configuration
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: str = "nl2q-schema-v2"
    
    # Authentication & Security
    jwt_secret_key: str = "your-secret-key-change-this"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 30
    oauth_providers: List[str] = []
    
    # Google OAuth
    google_oauth_client_id: Optional[str] = None
    google_oauth_client_secret: Optional[str] = None
    
    # GitHub OAuth
    github_oauth_client_id: Optional[str] = None
    github_oauth_client_secret: Optional[str] = None
    
    # Feature Flags
    enable_ml_suggestions: bool = True
    enable_real_time_collab: bool = True
    enable_multi_tenant: bool = True
    enable_query_caching: bool = True
    enable_streaming_responses: bool = True
    enable_advanced_analytics: bool = True
    
    # Monitoring
    sentry_dsn: Optional[str] = None
    enable_metrics: bool = True
    metrics_port: int = 8001
    log_level: str = "INFO"
    
    # Email & Notifications
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    from_email: Optional[str] = None
    
    # Storage
    storage_provider: str = "local"  # local, s3, gcs, azure
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_bucket_name: Optional[str] = None
    aws_region: str = "us-east-1"
    
    # ML/AI Configuration
    ml_model_path: str = "./models"
    enable_auto_ml: bool = False
    suggestion_model_version: str = "v2.1"
    analytics_model_version: str = "v1.5"
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst: int = 10
    
    # Security
    security_headers_enabled: bool = True
    https_only: bool = False
    
    # Multi-tenant Configuration
    default_tenant_id: str = "default"
    max_tenants: int = 100
    tenant_isolation_mode: str = "strict"  # strict, relaxed
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()