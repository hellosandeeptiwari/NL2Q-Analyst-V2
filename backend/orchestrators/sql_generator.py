"""
SQL Generation Module for Dynamic Agent Orchestrator
Handles sophisticated SQL generation with LLM intelligence
"""

import os
import json
from typing import Dict, List, Optional, Any
from openai import OpenAI


class SQLGenerator:
    """Handles sophisticated SQL generation using LLM with full metadata"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    async def generate_sql_with_retry(self, query: str, available_context: Dict[str, Any], 
                                    use_deterministic: bool = False, max_retries: int = 4) -> Dict[str, Any]:
        """Generate SQL with retry logic and enhanced error handling"""
        
        print(f"🔄 DEBUG: _generate_sql_with_retry called with use_deterministic={use_deterministic}")
        
        errors = []
        
        for attempt in range(1, max_retries + 1):
            print(f"🔄 SQL Generation Attempt {attempt}/{max_retries}")
            
            try:
                # Generate SQL using core method
                result = await self._generate_database_aware_sql_core(
                    query, available_context, use_deterministic, errors
                )
                
                if result.get("status") == "success":
                    return result
                else:
                    errors.append(f"Attempt {attempt}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                error_msg = f"SQL execution failed: SQL execution error: {str(e)}"
                errors.append(f"Attempt {attempt}: {error_msg}")
                print(f"❌ Attempt {attempt} failed: {error_msg}")
                
                if attempt < max_retries:
                    print(f"🔄 Retrying... ({attempt + 1}/{max_retries})")
                    print(f"📋 Error details: Exception: {error_msg[:100]}{'...' if len(error_msg) > 100 else ''}")
        
        print(f"❌ SQL generation failed after {max_retries} attempts")
        return {
            "status": "failed",
            "error": "All SQL generation attempts failed",
            "attempts": max_retries,
            "errors": errors
        }
    
    async def _generate_database_aware_sql_core(self, query: str, available_context: Dict[str, Any], 
                                              use_deterministic: bool = False, previous_errors: List[str] = None) -> Dict[str, Any]:
        """Core SQL generation with database awareness and LLM intelligence"""
        
        try:
            print(f"🧠 DEBUG: _generate_database_aware_sql_core called with use_deterministic={use_deterministic}")
            
            # Get database configuration
            db_name = os.getenv("SNOWFLAKE_DATABASE", "HEALTHCARE_PRICING_ANALYTICS_SAMPLE")
            schema_name = os.getenv("SNOWFLAKE_SCHEMA", "SAMPLES")
            db_engine = os.getenv("DB_ENGINE", "snowflake")
            
            # Get target tables from context
            target_tables = self._determine_target_tables(query, available_context)
            print(f"🎯 Target tables: {target_tables}")
            
            # Get complete table details from Pinecone with batched column metadata
            intelligent_context = await self.orchestrator._get_complete_table_details_from_pinecone(target_tables)
            
            # Create enhanced system prompt with LLM metadata instructions
            system_prompt = self._create_intelligent_sql_system_prompt(
                db_name, schema_name, "", target_tables, intelligent_context, 
                len(target_tables), query, use_deterministic, db_engine
            )
            
            # Prepare user prompt with error context if retrying
            user_prompt = f"Generate SQL for: {query}"
            if previous_errors:
                error_context = "\n".join(previous_errors[-2:])  # Last 2 errors
                user_prompt += f"\n\nPREVIOUS ERRORS TO AVOID:\n{error_context}"
            
            print(f"🔍 DEBUG: Using OpenAI model: gpt-4o-mini")
            
            # Generate SQL using LLM
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            sql_query = response.choices[0].message.content.strip()
            print(f"🔍 DEBUG: Raw LLM response: {sql_query[:100]}...")
            
            # Clean the SQL (remove markdown if present)
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            elif sql_query.startswith("```"):
                sql_query = sql_query[3:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            sql_query = sql_query.strip()
            
            # Apply database-specific fixes
            sql_query = self._apply_database_quoting(sql_query, target_tables, db_engine)
            
            print(f"🎯 DEBUG: Final cleaned SQL query: {sql_query}")
            if use_deterministic:
                print("🔧 DEBUG: This was generated using DETERMINISTIC mode!")
            
            # Test the SQL against database
            if self.orchestrator.db_connector:
                print(f"🧪 Testing generated SQL against database")
                test_result = self.orchestrator.db_connector.run(sql_query, dry_run=False)
                
                if test_result.error:
                    print(f"❌ SQL execution test failed: {test_result.error}")
                    return {
                        "status": "failed",
                        "error": f"SQL execution error: {test_result.error}",
                        "sql_query": sql_query
                    }
                else:
                    print(f"✅ SQL test passed - {len(test_result.rows) if test_result.rows else 0} rows")
            
            return {
                "sql_query": sql_query,
                "explanation": f"Database-aware SQL for {db_engine} generated from: {query}",
                "generation_method": "llm_with_batched_metadata",
                "status": "success"
            }
            
        except Exception as e:
            import traceback
            error_details = {
                "error": str(e),
                "error_type": type(e).__name__,
                "stack_trace": traceback.format_exc()
            }
            print(f"❌ Core SQL generation failed: {e}")
            return {"status": "failed", **error_details}
    
    def _determine_target_tables(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Determine target tables from context"""
        print(f"🔍 DEBUG: _determine_target_tables called with query: '{query}'")
        print(f"🔍 DEBUG: Context keys: {list(context.keys())}")
        
        # Look for matched_tables in context
        matched_tables = context.get('matched_tables', [])
        if matched_tables:
            print(f"🔍 DEBUG: Found matched_tables: {matched_tables}")
            return matched_tables
        
        # Fallback to schemas or other context
        schemas = context.get('schemas', {})
        if isinstance(schemas, dict) and schemas:
            tables = list(schemas.keys())
            print(f"🔍 DEBUG: Using schema tables: {tables}")
            return tables
        
        print(f"⚠️ No target tables found in context")
        return []
    
    def _create_intelligent_sql_system_prompt(self, db_name: str, schema_name: str, 
                                            schema_context: str, table_names: List[str],
                                            intelligent_context: Dict[str, Any], 
                                            schema_success_count: int, query: str,
                                            use_deterministic: bool = False, 
                                            db_engine: str = "snowflake") -> str:
        """Create enhanced system prompt with LLM intelligence and metadata instructions"""
        
        prompt_parts = [
            f"You are an AI-powered SQL generator with deep schema intelligence.",
            "",
            f"Database Context:",
            f"- Engine: {db_engine.upper().replace('-', ' ')}",
            f"- Database: {db_name}",
            f"- Schema: {schema_name}",
            "",
            "🚨 CRITICAL INSTRUCTIONS FOR LLM SQL GENERATION:",
            "• You MUST use the full metadata (all columns and types) provided for each table to inform your SQL generation.",
            "• Prefer multi-table joins and advanced SQL constructs when multiple tables are relevant to the user query.",
            "• Always use the correct database syntax as specified for the target engine (see syntax rules above).",
            "• Leverage column names, types, and any business context to select the best fields for joins, filters, and aggregations.",
            "• Do NOT guess or invent columns—use only those present in the metadata.",
            "• If a required field is missing, explain the limitation in a SQL comment.",
            "• Generate sophisticated, production-grade SQL, not just simple SELECTs.",
            "",
            f"🧠 AVAILABLE TABLES WITH COMPLETE METADATA:",
            f"You have {len(table_names)} pre-analyzed relevant tables: {', '.join(table_names)}",
            ""
        ]
        
        # Add table intelligence with batched column metadata
        table_insights = intelligent_context.get("table_insights", {})
        for table_name, insights in table_insights.items():
            prompt_parts.extend([
                f"📊 TABLE: {table_name}",
                ""
            ])
            
            # Add batched column information
            column_insights = insights.get('column_insights', [])
            if column_insights:
                prompt_parts.append("AVAILABLE COLUMNS:")
                for col in column_insights:
                    col_name = col['column_name']
                    data_type = col.get('data_type', 'unknown')
                    prompt_parts.append(f"- {col_name} ({data_type})")
                prompt_parts.append("")
        
        # Add database-specific syntax rules
        prompt_parts.extend([
            f"🚨 {db_engine.upper()} SYNTAX RULES:",
            self._get_database_syntax_rules(db_engine),
            "",
            f"User Query: {query}",
            "",
            f"Generate precise SQL that leverages ALL the column metadata above and STRICTLY follows {db_engine.upper()} syntax rules.",
            "Use sophisticated joins, aggregations, and filtering based on the available columns."
        ])
        
        return "\n".join(prompt_parts)
    
    def _get_database_syntax_rules(self, db_engine: str) -> str:
        """Get database-specific syntax rules"""
        if db_engine == "snowflake":
            return """- Use double quotes for identifiers: "TABLE"."COLUMN"
- Use LIMIT n for row limiting
- Use proper Snowflake functions and syntax"""
        elif db_engine == "azure_sql":
            return """- Use square brackets for identifiers: [TABLE].[COLUMN]
- Use TOP n for row limiting  
- Use Azure SQL Server functions and syntax"""
        else:
            return "- Use standard SQL syntax"
    
    def _apply_database_quoting(self, sql_query: str, table_names: List[str], db_engine: str) -> str:
        """Apply proper database-specific quoting"""
        # This would contain the database-specific quoting logic
        # For now, return as-is
        return sql_query