"""
Multi-LLM Service for NL2Q Analyst V2

Supports multiple LLM providers with automatic selection and fallback.
"""
import asyncio
import json
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from enum import Enum
import structlog

from src.core.config import settings
from src.core.exceptions import LLMProviderError

logger = structlog.get_logger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate_sql(
        self,
        natural_language: str,
        schema_context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate SQL from natural language query."""
        pass
    
    @abstractmethod
    async def generate_insights(
        self,
        data_summary: Dict[str, Any],
        query_context: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate insights from data."""
        pass
    
    @abstractmethod
    async def ping(self) -> bool:
        """Check if the LLM provider is available."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT client."""
    
    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        except ImportError:
            raise ImportError("openai package not installed")
    
    async def generate_sql(
        self,
        natural_language: str,
        schema_context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate SQL using OpenAI GPT."""
        try:
            model = kwargs.get("model", "gpt-4-turbo-preview")
            
            system_prompt = self._build_sql_system_prompt(schema_context)
            user_prompt = f"Convert this natural language query to SQL: {natural_language}"
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            sql_response = response.choices[0].message.content
            
            return {
                "sql": self._extract_sql(sql_response),
                "explanation": sql_response,
                "provider": "openai",
                "model": model,
                "confidence": 0.9  # Could be enhanced with actual confidence scoring
            }
            
        except Exception as e:
            logger.error("OpenAI SQL generation failed", error=str(e))
            raise LLMProviderError(f"OpenAI error: {str(e)}", provider="openai")
    
    async def generate_insights(
        self,
        data_summary: Dict[str, Any],
        query_context: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate insights using OpenAI GPT."""
        try:
            model = kwargs.get("model", "gpt-4-turbo-preview")
            
            system_prompt = """You are a data analyst AI. Analyze the provided data summary and generate meaningful insights, trends, and recommendations."""
            
            user_prompt = f"""
            Context: {query_context}
            Data Summary: {json.dumps(data_summary, indent=2)}
            
            Please provide:
            1. Key insights from the data
            2. Notable trends or patterns
            3. Actionable recommendations
            4. Statistical observations
            """
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            insights = response.choices[0].message.content
            
            return {
                "insights": insights,
                "provider": "openai",
                "model": model,
                "data_points_analyzed": len(data_summary.get("rows", []))
            }
            
        except Exception as e:
            logger.error("OpenAI insights generation failed", error=str(e))
            raise LLMProviderError(f"OpenAI error: {str(e)}", provider="openai")
    
    async def ping(self) -> bool:
        """Check OpenAI API availability."""
        try:
            response = await self.client.models.list()
            return True
        except Exception:
            return False
    
    def _build_sql_system_prompt(self, schema_context: Dict[str, Any]) -> str:
        """Build system prompt for SQL generation."""
        return f"""You are an expert SQL generator. Convert natural language queries to SQL based on the provided database schema.

Schema Context:
{json.dumps(schema_context, indent=2)}

Guidelines:
1. Generate clean, efficient SQL queries
2. Use proper table and column names from the schema
3. Add appropriate WHERE clauses for filtering
4. Include LIMIT clauses for large result sets (default: 100)
5. Use proper JOINs when needed
6. Return only the SQL query in your response
7. Ensure the query is safe and doesn't modify data

Response format: Provide the SQL query and a brief explanation."""
    
    def _extract_sql(self, response: str) -> str:
        """Extract SQL query from LLM response."""
        # Look for SQL code blocks
        if "```sql" in response:
            start = response.find("```sql") + 6
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        # Look for SQL keywords to extract query
        sql_keywords = ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE"]
        lines = response.split("\n")
        
        for i, line in enumerate(lines):
            if any(keyword in line.upper() for keyword in sql_keywords):
                # Found SQL, extract until end or next non-SQL line
                sql_lines = []
                for j in range(i, len(lines)):
                    if lines[j].strip():
                        sql_lines.append(lines[j])
                    elif sql_lines:  # Empty line after SQL content
                        break
                return "\n".join(sql_lines)
        
        return response.strip()


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude client."""
    
    def __init__(self):
        if not settings.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")
        
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        except ImportError:
            raise ImportError("anthropic package not installed")
    
    async def generate_sql(self, natural_language: str, schema_context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate SQL using Claude."""
        try:
            model = kwargs.get("model", "claude-3-sonnet-20240229")
            
            system_prompt = self._build_sql_system_prompt(schema_context)
            user_prompt = f"Convert this natural language query to SQL: {natural_language}"
            
            response = await self.client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=0.1,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            sql_response = response.content[0].text
            
            return {
                "sql": self._extract_sql(sql_response),
                "explanation": sql_response,
                "provider": "anthropic",
                "model": model,
                "confidence": 0.9
            }
            
        except Exception as e:
            logger.error("Anthropic SQL generation failed", error=str(e))
            raise LLMProviderError(f"Anthropic error: {str(e)}", provider="anthropic")
    
    async def generate_insights(self, data_summary: Dict[str, Any], query_context: str, **kwargs) -> Dict[str, Any]:
        """Generate insights using Claude."""
        try:
            model = kwargs.get("model", "claude-3-sonnet-20240229")
            
            system_prompt = "You are a data analyst AI specializing in generating actionable insights from data."
            
            user_prompt = f"""
            Analyze this data and provide insights:
            
            Context: {query_context}
            Data: {json.dumps(data_summary, indent=2)}
            
            Provide key insights, trends, and recommendations.
            """
            
            response = await self.client.messages.create(
                model=model,
                max_tokens=1500,
                temperature=0.3,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            insights = response.content[0].text
            
            return {
                "insights": insights,
                "provider": "anthropic",
                "model": model,
                "data_points_analyzed": len(data_summary.get("rows", []))
            }
            
        except Exception as e:
            logger.error("Anthropic insights generation failed", error=str(e))
            raise LLMProviderError(f"Anthropic error: {str(e)}", provider="anthropic")
    
    async def ping(self) -> bool:
        """Check Anthropic API availability."""
        try:
            # Simple message to test API
            response = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}]
            )
            return True
        except Exception:
            return False
    
    def _build_sql_system_prompt(self, schema_context: Dict[str, Any]) -> str:
        """Build system prompt for SQL generation."""
        return f"""Generate SQL queries from natural language based on this schema:

{json.dumps(schema_context, indent=2)}

Rules:
- Generate safe, read-only queries
- Use proper table/column names
- Add LIMIT clauses (default 100)
- Provide clean, efficient SQL"""
    
    def _extract_sql(self, response: str) -> str:
        """Extract SQL from Claude response."""
        # Similar logic to OpenAI
        if "```sql" in response:
            start = response.find("```sql") + 6
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        return response.strip()


class LLMService:
    """Multi-LLM service with provider selection and fallback."""
    
    def __init__(self):
        self.providers: Dict[LLMProvider, BaseLLMClient] = {}
        self.default_provider = LLMProvider(settings.default_llm_provider)
        self._initialized = False
    
    async def initialize(self):
        """Initialize available LLM providers."""
        if self._initialized:
            return
        
        logger.info("Initializing LLM service")
        
        # Initialize OpenAI if configured
        if settings.openai_api_key:
            try:
                self.providers[LLMProvider.OPENAI] = OpenAIClient()
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning("Failed to initialize OpenAI client", error=str(e))
        
        # Initialize Anthropic if configured
        if settings.anthropic_api_key:
            try:
                self.providers[LLMProvider.ANTHROPIC] = AnthropicClient()
                logger.info("Anthropic client initialized")
            except Exception as e:
                logger.warning("Failed to initialize Anthropic client", error=str(e))
        
        # Initialize Google if configured
        if settings.google_api_key:
            try:
                # Google client implementation would go here
                logger.info("Google client not implemented yet")
            except Exception as e:
                logger.warning("Failed to initialize Google client", error=str(e))
        
        if not self.providers:
            raise ValueError("No LLM providers configured")
        
        self._initialized = True
        logger.info("LLM service initialized", providers=list(self.providers.keys()))
    
    async def generate_sql(
        self,
        natural_language: str,
        schema_context: Dict[str, Any],
        provider: Optional[LLMProvider] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate SQL with provider selection and fallback."""
        providers_to_try = [provider] if provider else [self.default_provider]
        
        # Add fallback providers
        for p in self.providers:
            if p not in providers_to_try:
                providers_to_try.append(p)
        
        last_error = None
        
        for provider in providers_to_try:
            if provider not in self.providers:
                continue
                
            try:
                logger.info("Attempting SQL generation", provider=provider.value)
                result = await self.providers[provider].generate_sql(
                    natural_language, schema_context, **kwargs
                )
                logger.info("SQL generation successful", provider=provider.value)
                return result
                
            except Exception as e:
                logger.warning("SQL generation failed", provider=provider.value, error=str(e))
                last_error = e
                continue
        
        raise LLMProviderError(
            f"All LLM providers failed. Last error: {str(last_error)}",
            details={"tried_providers": [p.value for p in providers_to_try]}
        )
    
    async def generate_insights(
        self,
        data_summary: Dict[str, Any],
        query_context: str,
        provider: Optional[LLMProvider] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate insights with provider selection and fallback."""
        providers_to_try = [provider] if provider else [self.default_provider]
        
        for p in self.providers:
            if p not in providers_to_try:
                providers_to_try.append(p)
        
        last_error = None
        
        for provider in providers_to_try:
            if provider not in self.providers:
                continue
                
            try:
                logger.info("Attempting insights generation", provider=provider.value)
                result = await self.providers[provider].generate_insights(
                    data_summary, query_context, **kwargs
                )
                logger.info("Insights generation successful", provider=provider.value)
                return result
                
            except Exception as e:
                logger.warning("Insights generation failed", provider=provider.value, error=str(e))
                last_error = e
                continue
        
        raise LLMProviderError(
            f"All LLM providers failed. Last error: {str(last_error)}",
            details={"tried_providers": [p.value for p in providers_to_try]}
        )
    
    async def ping(self) -> Dict[str, bool]:
        """Check availability of all providers."""
        status = {}
        
        for provider, client in self.providers.items():
            try:
                status[provider.value] = await client.ping()
            except Exception:
                status[provider.value] = False
        
        return status
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up LLM service")
        self.providers.clear()
        self._initialized = False
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return [provider.value for provider in self.providers.keys()]


# Global LLM service instance
llm_service = LLMService()