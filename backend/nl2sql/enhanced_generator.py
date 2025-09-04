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

def generate_enhanced_sql(natural_language: str, schema_context: dict, database_type: str = "snowflake", limit: int = 10) -> dict:
    """Generate SQL using enhanced schema with vector grounding context"""
    
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

# Legacy function for backward compatibility
def generate_sql(natural_language: str, schema_snapshot: dict, constraints: GuardrailConfig) -> GeneratedSQL:
    """Legacy function - will use enhanced schema if available"""
    
    # Check if this is an enhanced schema
    if isinstance(schema_snapshot, dict) and 'schema_version' in schema_snapshot:
        return generate_sql_with_enhanced_schema(natural_language, schema_snapshot, constraints)
    
    # Fallback to simple schema format
    simple_prompt = f"""
You are an expert SQL generator. Generate a safe SELECT query for the following request.

Schema: {schema_snapshot}
Request: {natural_language}

Constraints:
- Only SELECT statements
- Add LIMIT clause if unbounded
- Use exact table/column names from schema

SQL:"""
    
    try:
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": simple_prompt}],
            max_completion_tokens=500,
            temperature=0.1
        )
        
        sql = response.choices[0].message.content.strip()
        safe_sql, added_limit = sanitize_sql(sql, constraints)
        suggestions = generate_query_suggestions(natural_language, schema_snapshot)
        
        return GeneratedSQL(
            sql=safe_sql,
            rationale="Generated with basic schema",
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
