"""
Enhanced SQL Generator with Rich Schema Context
Supports healthcare/life-sciences domain with governance and semantic understanding
"""

import os
import openai
from dataclasses import dataclass
from typing import List, Dict, Optional
from .guardrails import GuardrailConfig, sanitize_sql
from ..db.enhanced_schema import format_schema_for_llm

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

@dataclass
class GeneratedSQL:
    sql: str
    rationale: str
    added_limit: bool
    suggestions: List[str]
    confidence_score: Optional[float] = None
    semantic_matches: Optional[List[str]] = None

# Enhanced few-shot examples with healthcare context
HEALTHCARE_EXAMPLES = [
    {
        "nl": "Show me the top 5 prescribed medications by volume",
        "sql": "SELECT product_name, COUNT(*) as prescription_count FROM rx_facts rf JOIN product_master pm ON rf.product_id = pm.product_id GROUP BY product_name ORDER BY prescription_count DESC LIMIT 5",
        "context": "Aggregated product analysis with proper joins"
    },
    {
        "nl": "Find new patient starts for diabetes medications in Q1 2024",
        "sql": "SELECT COUNT(DISTINCT patient_hash) as new_patients FROM rx_facts rf JOIN product_master pm ON rf.product_id = pm.product_id WHERE pm.therapeutic_area = 'DIABETES' AND rf.rx_type = 'NEW_START' AND rf.fill_date BETWEEN '2024-01-01' AND '2024-03-31'",
        "context": "KPI calculation with date filtering and therapeutic area"
    },
    {
        "nl": "Show physician prescribing patterns by specialty",
        "sql": "SELECT hm.specialty, COUNT(*) as rx_count, COUNT(DISTINCT rf.patient_hash) as unique_patients FROM rx_facts rf JOIN hcp_master hm ON rf.hcp_id = hm.hcp_id GROUP BY hm.specialty ORDER BY rx_count DESC LIMIT 10",
        "context": "HCP analysis with aggregation guardrails"
    }
]

ENHANCED_SYSTEM_PROMPT = """
You are an expert SQL analyst specialized in healthcare/life-sciences data analysis. You generate safe, compliant SQL queries for read-only analytics.

CORE CAPABILITIES:
- Generate sophisticated analytical queries for healthcare data
- Understand pharmaceutical/medical terminology and metrics  
- Apply data governance and privacy protection automatically
- Use semantic understanding of business concepts and KPIs
- Optimize queries for performance on large datasets

CRITICAL CONSTRAINTS:
- ONLY SELECT statements - never CREATE, INSERT, UPDATE, DELETE, DROP, ALTER
- Always enforce small-cell suppression (>=11 patients/records)
- Never expose PHI/PII directly - use aggregations and safe geographic levels
- Respect data governance flags and sensitivity labels
- Add LIMIT clauses to prevent large result sets
- Use proper case-sensitive table/column names from schema

HEALTHCARE-SPECIFIC GUIDELINES:
- For patient counts: Always use COUNT(DISTINCT patient_hash/patient_id)
- Geographic analysis: Prefer ZIP3/DMA over ZIP5 for privacy
- Date ranges: Use fill_date/event_date for temporal analysis
- Therapeutic areas: Use standardized TA classifications from product hierarchy
- HCP analysis: Aggregate by specialty/segment, not individual providers
- Prescription metrics: Distinguish NEW_START vs REFILL for NPS calculations

QUERY OPTIMIZATION:
- Use partitioned columns (date fields) in WHERE clauses
- Prefer indexed/clustered columns for joins
- Sample large tables when doing exploratory analysis
- Use materialized views when available for complex joins

SEMANTIC UNDERSTANDING:
- "New patients" = NEW_START prescriptions, COUNT(DISTINCT patient_hash)
- "Persistence" = Patients with fills spanning 6+ months
- "Market share" = Product/brand % of total category volume
- "Writers" = COUNT(DISTINCT hcp_id) with prescriptions
- "Therapy area" = Therapeutic classification from product master

Here are domain-specific examples:
""" + "\n".join([f"Business Question: {ex['nl']}\nSQL Solution: {ex['sql']}\nContext: {ex['context']}\n" for ex in HEALTHCARE_EXAMPLES])

SCHEMA_CONTEXT_PROMPT = """
DATABASE-SPECIFIC SYNTAX & DOCUMENTATION:
{database_documentation}

SCHEMA INTELLIGENCE:
Use the following rich schema metadata to generate accurate queries. Pay attention to:
- Semantic roles (date.event, measure.quantity, entity.id, etc.)
- Business aliases and natural language mappings
- Value ranges and categorical options
- Relationship hints and join patterns
- Governance flags and privacy constraints
- Database-specific syntax and functions from documentation above

Schema Snapshot:
{schema_context}

GUARDRAILS ACTIVE:
- Default row limit: {default_limit}
- Small cell threshold: {small_cell_threshold}
- Sensitive data protection enabled
- Performance optimization recommended
- Use database-specific syntax from documentation above

"""

REFINEMENT_PROMPT = """
Generate 2-3 analytical variations of the given healthcare/business question. Focus on:
1. Different time windows or date ranges
2. Alternative aggregation levels (patient vs HCP vs geography)
3. Related business metrics or KPI perspectives
4. Drill-down or roll-up variations

Ensure all suggestions respect data governance and privacy constraints.
"""

def generate_query_suggestions(natural_language: str, schema_snapshot: dict = None) -> List[str]:
    """Generate alternative query suggestions with minimal context to prevent token overload"""
    # Use minimal schema info to prevent massive token usage
    schema_info = "Available tables: limited subset"
    
    prompt = f"Generate 2-3 alternative questions for: {natural_language}"
    
    try:
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=150,
            temperature=0.7
        )
        suggestions_text = response.choices[0].message.content.strip()
        # Parse into list
        suggestions = [s.strip().lstrip('123.-') for s in suggestions_text.split('\n') if s.strip()]
        return suggestions[:3]
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        return []

def generate_enhanced_sql(natural_language: str, schema_context: dict, database_type: str = None, limit: int = 10) -> dict:
    """Generate SQL using enhanced schema with vector grounding context"""
    
    # Get database type from environment if not provided
    if database_type is None:
        database_type = os.getenv("DB_ENGINE", "azure_sql")
    
    try:
        # Extract vector insights
        vector_insights = schema_context.get('vector_insights', {})
        tables_info = schema_context.get('tables', {})
        
        # Build enhanced prompt with vector context
        enhanced_prompt = f"""
{ENHANCED_SYSTEM_PROMPT}

VECTOR SEARCH CONTEXT:
- Query Intent: {vector_insights.get('query_intent', natural_language)}
- AI Confidence: {vector_insights.get('confidence_score', 0.8):.1%}
- Selected Table: {vector_insights.get('selected_table', 'unknown')}
- Analysis Type: {vector_insights.get('analysis_type', 'general')}

VALIDATED SCHEMA INFORMATION:
"""
        
        # Add table information
        for table_name, table_info in tables_info.items():
            enhanced_prompt += f"""
Table: {table_name}
Description: {table_info.get('description', 'No description')}
Available Columns: {', '.join(table_info.get('columns', []))}
"""
            
            # Add vector-validated column matches
            validated_cols = table_info.get('validated_matches', [])
            if validated_cols:
                enhanced_prompt += f"Vector-Matched Columns (with confidence):\n"
                for col in validated_cols:
                    enhanced_prompt += f"  - {col.get('column_name', 'unknown')}: {col.get('confidence', 0):.1%} confidence\n"
        
        enhanced_prompt += f"""

GENERATION GUIDELINES:
- Use case-sensitive column/table names with quotes: "{vector_insights.get('selected_table', 'TABLE_NAME')}"
- Focus on the vector-matched high-confidence columns
- Generate {vector_insights.get('analysis_type', 'analysis')} appropriate for the query intent
- Limit results to {limit} rows for performance
- Use Snowflake-compatible SQL syntax

User Query: {natural_language}

Generate the SQL query:"""

        # Call GPT-4o-mini
        import openai
        import os
        
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "user", "content": enhanced_prompt}
            ],
            max_completion_tokens=800,
            temperature=0.1
        )
        
        sql = response.choices[0].message.content.strip()
        
        # Clean up SQL formatting
        if "```sql" in sql:
            sql = sql.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql:
            sql = sql.split("```")[1].strip()
        
        return {
            "sql": sql,
            "confidence": vector_insights.get('confidence_score', 0.8),
            "analysis_type": vector_insights.get('analysis_type', 'general'),
            "vector_grounded": True,
            "model_used": "gpt-4o-mini",
            "table_used": vector_insights.get('selected_table'),
            "matched_columns": [col.get('column_name') for col in vector_insights.get('validated_columns', [])]
        }
        
    except Exception as e:
        print(f"❌ Enhanced SQL generation failed: {e}")
        return None

    """Generate SQL using enhanced schema with rich metadata"""
    
    # Format schema for LLM
    schema_context = format_schema_for_llm(enhanced_schema)
    
    # Extract database documentation if available
    database_documentation = enhanced_schema.get('database_documentation', 'Standard SQL - No specific database documentation available')
    
    # Build comprehensive prompt with schema context and database documentation
    full_prompt = ENHANCED_SYSTEM_PROMPT + "\n\n" + SCHEMA_CONTEXT_PROMPT.format(
        database_documentation=database_documentation,
        schema_context=schema_context,
        default_limit=constraints.default_limit,
        small_cell_threshold=enhanced_schema.get('small_cell_threshold', 11)
    )
    
    user_prompt = f"Business Question: {natural_language}\n\nGenerate SQL query:"
    
    try:
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=1000,
            temperature=0.1  # Lower temperature for more consistent SQL
        )
        
        sql = response.choices[0].message.content.strip()
        
        # Clean up SQL (remove markdown formatting if present)
        if sql.startswith("```sql"):
            sql = sql[6:]
        if sql.endswith("```"):
            sql = sql[:-3]
        sql = sql.strip()
        
        # Apply guardrails
        safe_sql, added_limit = sanitize_sql(sql, constraints)
        
        # Generate suggestions
        suggestions = generate_query_suggestions(natural_language, enhanced_schema)
        
        # Build enhanced rationale
        rationale = f"Generated using enhanced schema with {len(enhanced_schema.get('tables', []))} tables. Applied healthcare-specific governance and optimization patterns."
        
        return GeneratedSQL(
            sql=safe_sql,
            rationale=rationale,
            added_limit=added_limit,
            suggestions=suggestions,
            confidence_score=0.85  # High confidence with enhanced schema
        )
        
    except Exception as e:
        print(f"Error generating SQL: {e}")
        return GeneratedSQL(
            sql="SELECT 'Error generating query' as error",
            rationale=f"Failed to generate SQL: {str(e)}",
            added_limit=False,
            suggestions=[],
            confidence_score=0.0
        )

# Legacy function for backward compatibility with new deterministic option
def generate_sql(natural_language: str, schema_snapshot: dict, constraints: GuardrailConfig, 
                use_deterministic: bool = False) -> GeneratedSQL:
    """Enhanced function with error-aware retry capability and optional deterministic mode"""
    
    # Import required modules at function start to avoid UnboundLocalError
    import sys
    import os
    
    # Get database engine from environment - defaults to azure_sql
    db_engine = os.getenv("DB_ENGINE", "azure_sql").lower()
    
    # Get database-specific syntax rules
    def get_db_syntax_rules(db_engine: str) -> str:
        """Return database-specific SQL syntax rules"""
        if db_engine in ["azure_sql", "mssql", "sqlserver"]:
            return """
DATABASE: Microsoft Azure SQL Server / SQL Server
SYNTAX RULES:
- Use square brackets [table].[column] for identifiers with spaces
- Use TOP N instead of LIMIT N
- Single SELECT statement only (no semicolons or multiple statements)
- Use ISNULL() instead of COALESCE() when possible
- Date format: 'YYYY-MM-DD' for dates
- String concatenation with + operator
- Use dbo schema prefix: dbo.TableName"""
        elif db_engine == "snowflake":
            return """
DATABASE: Snowflake
SYNTAX RULES:
- Case-insensitive identifiers
- Use LIMIT N for row limiting
- Single SELECT statement only
- Use || for string concatenation
- Date format: 'YYYY-MM-DD' for dates"""
        elif db_engine in ["postgres", "postgresql"]:
            return """
DATABASE: PostgreSQL
SYNTAX RULES:
- Case-sensitive identifiers (use quotes if needed)
- Use LIMIT N for row limiting
- Single SELECT statement only
- Use || for string concatenation
- Date format: 'YYYY-MM-DD' for dates"""
        else:
            return """
DATABASE: Generic SQL
SYNTAX RULES:
- Standard SQL syntax
- Single SELECT statement only
- Use LIMIT N for row limiting"""
    
    db_syntax_rules = get_db_syntax_rules(db_engine)
    
    # Option 1: Use enhanced deterministic column-first approach with micro-patches
    if use_deterministic and isinstance(schema_snapshot, dict) and 'tables' in schema_snapshot:
        try:
            # Import the enhanced prompt functions
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from enhanced_prompt_with_micro_patches import create_production_prompt
            
            # Convert schema to catalog format
            catalog_tables = [{'name': table} for table in schema_snapshot.keys()]
            catalog_columns = []
            
            # Convert columns format
            for table_name, columns in schema_snapshot.items():
                if isinstance(columns, list):
                    for col_name in columns:
                        catalog_columns.append({
                            'table_name': table_name,
                            'column_name': col_name,
                            'data_type': 'VARCHAR'  # Default type if not specified
                        })
            
            catalog_keys = {
                'primary': {},
                'foreign': []
            }
            
            # Generate enhanced prompt with micro-patches  
            # Map our db_engine to the dialect expected by create_production_prompt
            dialect_map = {
                "azure_sql": "mssql",
                "mssql": "mssql", 
                "sqlserver": "mssql",
                "snowflake": "snowflake",
                "postgres": "postgresql",
                "postgresql": "postgresql"
            }
            dialect = dialect_map.get(db_engine, "mssql")  # Default to mssql for Azure SQL
            
            enhanced_prompt = create_production_prompt(
                user_query=natural_language,
                db_tables=catalog_tables,
                db_columns=catalog_columns,
                db_keys=catalog_keys,
                dialect=dialect
            )
            
            # Call LLM with enhanced prompt
            response = openai.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": enhanced_prompt}],
                max_completion_tokens=1000,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse response - expect JSON plan first, then SQL
            if 'SQL:' in response_text:
                sql_part = response_text.split('SQL:')[-1].strip()
            else:
                sql_part = response_text
            
            # Clean SQL
            if sql_part.startswith("```sql"):
                sql_part = sql_part[6:]
            elif sql_part.startswith("```"):
                sql_part = sql_part[3:]
            if sql_part.endswith("```"):
                sql_part = sql_part[:-3]
            
            sql_part = sql_part.strip()
            
            # Apply guardrails
            safe_sql, added_limit = sanitize_sql(sql_part, constraints)
            
            return GeneratedSQL(
                sql=safe_sql,
                rationale="Generated with enhanced deterministic approach using column-first selection and micro-patches",
                added_limit=added_limit,
                suggestions=generate_query_suggestions(natural_language, schema_snapshot),
                confidence_score=0.95  # High confidence with deterministic approach
            )
            
        except ImportError:
            print("⚠️ Enhanced deterministic prompt not available, falling back to basic deterministic mode")
        except Exception as e:
            print(f"⚠️ Enhanced deterministic generation failed: {e}, falling back to enhanced mode")
    
    # Option 2: Check if this is an enhanced schema with error context
    if isinstance(schema_snapshot, dict) and ('error_context' in schema_snapshot or 'retry_attempt' in schema_snapshot):
        return generate_sql_with_error_feedback(natural_language, schema_snapshot, constraints)
    
    # Option 3: Check if this is an enhanced schema
    if isinstance(schema_snapshot, dict) and 'schema_version' in schema_snapshot:
        return generate_sql_with_enhanced_schema(natural_language, schema_snapshot, constraints)
    
    # Option 4: Fallback to improved simple schema format with deterministic principles
    simple_prompt = f"""
Role: Expert SQL generator with deterministic approach.

{db_syntax_rules}

AUTHORITATIVE CATALOG:
{format_catalog_for_prompt(schema_snapshot)}

CRITICAL RULES:
1. Use ONLY columns/tables present in catalog above
2. If needed concept can't be mapped, explain why
3. Generate SINGLE SELECT statement only (no semicolons, no multiple statements)
4. For Azure SQL Server: Use TOP N instead of LIMIT N
5. Use exact table/column names from catalog
6. Follow database-specific syntax rules above

REQUEST: {natural_language}

Generate a single SQL SELECT statement or explain if concepts cannot be mapped:"""
    
    try:
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": simple_prompt}],
            max_completion_tokens=500,
            temperature=0.1
        )
        
        sql = response.choices[0].message.content.strip()
        
        # Check if response is an explanation rather than SQL
        if "cannot be mapped" in sql.lower() or "not found" in sql.lower():
            return GeneratedSQL(
                sql="SELECT 'Concept mapping failed' as error",
                rationale=sql,
                added_limit=False,
                suggestions=generate_query_suggestions(natural_language, schema_snapshot)
            )
        
        safe_sql, added_limit = sanitize_sql(sql, constraints)
        suggestions = generate_query_suggestions(natural_language, schema_snapshot)
        
        return GeneratedSQL(
            sql=safe_sql,
            rationale="Generated with improved deterministic principles",
            added_limit=added_limit,
            suggestions=suggestions
        )
        
    except Exception as e:
        return GeneratedSQL(
            sql="SELECT 'Error' as error",
            rationale=f"Error: {str(e)}",
            added_limit=False,
            suggestions=[]
        )

def format_catalog_for_prompt(schema_snapshot: dict) -> str:
    """Format schema in catalog-like structure for deterministic prompting with clear table-column organization"""
    if not schema_snapshot:
        return "No catalog provided"
    
    catalog_text = "AVAILABLE TABLES AND COLUMNS:\n" + "="*60 + "\n\n"
    
    for table_name, columns in schema_snapshot.items():
        catalog_text += f"📊 TABLE: dbo.{table_name}\n"
        catalog_text += "-" * (len(table_name) + 15) + "\n"
        
        if isinstance(columns, list):
            # Organize columns by category for better understanding
            key_columns = []
            id_columns = []
            product_columns = []
            metric_columns = []
            other_columns = []
            
            for col in columns:
                col_lower = col.lower()
                if 'id' in col_lower and col_lower.endswith('id'):
                    id_columns.append(col)
                elif any(word in col_lower for word in ['product', 'tirosint', 'licart', 'flector']):
                    product_columns.append(col) 
                elif any(word in col_lower for word in ['trx', 'nrx', 'tqty', 'nqty', 'share', 'calls', 'samples']):
                    metric_columns.append(col)
                elif any(word in col_lower for word in ['name', 'territory', 'region', 'prescriber', 'address', 'city', 'state']):
                    key_columns.append(col)
                else:
                    other_columns.append(col)
            
            if key_columns:
                catalog_text += f"  🏷️  Key Columns: {', '.join(key_columns)}\n"
            if id_columns:
                catalog_text += f"  🔑 ID Columns: {', '.join(id_columns)}\n"
            if product_columns:
                catalog_text += f"  💊 Product Columns: {', '.join(product_columns)}\n"
            if metric_columns:
                catalog_text += f"  📈 Metrics: {', '.join(metric_columns)}\n"
            if other_columns:
                catalog_text += f"  📋 Other: {', '.join(other_columns)}\n"
        else:
            catalog_text += f"  Columns: {columns}\n"
        
        catalog_text += "\n"
    
    catalog_text += "⚠️  CRITICAL: Only use columns from the exact table they belong to above!\n"
    catalog_text += "⚠️  Do NOT assume columns exist in tables where they are not listed!\n"
    
    return catalog_text

def generate_sql_with_error_feedback(natural_language: str, enhanced_schema: dict, constraints: GuardrailConfig) -> GeneratedSQL:
    """Generate SQL with error context feedback for iterative improvement"""
    
    try:
        # Extract components
        tables = enhanced_schema.get('tables', {})
        error_context = enhanced_schema.get('error_context', '')
        retry_attempt = enhanced_schema.get('retry_attempt', 1)
        database_type = enhanced_schema.get('database_type', 'snowflake')
        
        # Build schema text
        schema_text = "AVAILABLE TABLES AND COLUMNS:\n"
        for table_name, columns in tables.items():
            schema_text += f"Table: {table_name}\n"
            schema_text += f"Columns: {', '.join(columns)}\n\n"
        
        # Enhanced error-aware system prompt
        system_prompt = f"""You are an expert Snowflake SQL generator with error correction capabilities.

DATABASE CONTEXT:
- Engine: {database_type.upper()}
- Current Retry: Attempt {retry_attempt}
- Previous attempts had errors that need fixing

CRITICAL SNOWFLAKE RULES:
1. Use double quotes for mixed-case identifiers: "Table_Name", "Column_Name"
2. CTE column aliases must be consistently quoted: WITH cte ("alias1", "alias2") AS (...)
3. NEVER mix window functions with GROUP BY in same query level
4. Use subqueries/CTEs when combining aggregates with window functions
5. Always validate column references match available schema
6. Use proper Snowflake syntax for all functions

ERROR CORRECTION PRIORITY:
- Fix compilation errors (invalid identifiers, syntax issues)
- Ensure column aliases are properly quoted in CTEs
- Validate all table and column names exist in schema
- Use appropriate Snowflake functions and syntax

{schema_text}

Generate syntactically correct Snowflake SQL that will execute without errors."""

        user_prompt = f"""Request: {natural_language}

{error_context}

Generate corrected SQL that fixes all identified issues:"""

        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=800,
            temperature=0.1  # Lower temperature for more consistent fixes
        )
        
        sql = response.choices[0].message.content.strip()
        
        # Clean up SQL formatting
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        sql = sql.strip()
        
        # Apply guardrails
        safe_sql, added_limit = sanitize_sql(sql, constraints)
        
        # Generate suggestions
        suggestions = generate_query_suggestions(natural_language, tables)
        
        # Build rationale
        rationale = f"Generated with error feedback (attempt {retry_attempt}). Applied Snowflake-specific error corrections."
        
        return GeneratedSQL(
            sql=safe_sql,
            rationale=rationale,
            added_limit=added_limit,
            suggestions=suggestions,
            confidence_score=0.9 - (retry_attempt * 0.1)  # Decrease confidence with retries
        )
        
    except Exception as e:
        print(f"Error in error-aware SQL generation: {e}")
        return GeneratedSQL(
            sql="SELECT 'Error in retry generation' as error",
            rationale=f"Failed to generate SQL with error feedback: {str(e)}",
            added_limit=False,
            suggestions=[],
            confidence_score=0.0
        )
