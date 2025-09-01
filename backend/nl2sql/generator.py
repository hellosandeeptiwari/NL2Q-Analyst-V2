from dataclasses import dataclass
import openai
import os
from backend.nl2sql.guardrails import sanitize_sql, GuardrailConfig

@dataclass
class GeneratedSQL:
    sql: str
    rationale: str
    added_limit: bool
    suggestions: list[str]  # New: list of suggested rephrased queries

FEW_SHOTS = [
    {
        "nl": "Show all orders from last month.",
        "sql": "SELECT * FROM orders WHERE order_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')"
    },
    {
        "nl": "List customers who spent more than $1000.",
        "sql": "SELECT customer_id, SUM(amount) FROM transactions GROUP BY customer_id HAVING SUM(amount) > 1000"
    },
    {
        "nl": "Get product sales by category.",
        "sql": "SELECT category, SUM(sales) FROM products GROUP BY category"
    },
    {
        "nl": "Show daily active users for the past week.",
        "sql": "SELECT date, COUNT(DISTINCT user_id) FROM user_activity WHERE date >= CURRENT_DATE - INTERVAL '7 days' GROUP BY date"
    },
    {
        "nl": "Find top 5 selling products.",
        "sql": "SELECT product_id, SUM(sales) FROM sales GROUP BY product_id ORDER BY SUM(sales) DESC LIMIT 5"
    },
    {
        "nl": "Show average order value by region.",
        "sql": "SELECT region, AVG(order_value) FROM orders GROUP BY region"
    }
]

SYSTEM_PROMPT = """
You are an expert SQL generator for read-only database operations. Given a database schema and a natural language question, generate a safe, single-statement SQL SELECT query only.

CRITICAL CONSTRAINTS:
- ONLY generate SELECT statements - never CREATE, INSERT, UPDATE, DELETE, DROP, ALTER, or any DDL/DML operations
- NEVER create tables, stored procedures, functions, views, or any database objects
- ONLY read data using SELECT queries
- Always add a LIMIT if the query is unbounded to prevent large result sets
- Use the provided schema information to generate accurate queries

Database Usage Policy:
- Read-only access only
- No schema modifications allowed
- No data modifications allowed
- Safe query generation with guardrails

SNOWFLAKE-SPECIFIC GUIDELINES:
- Snowflake is case-sensitive for object names (tables, columns, etc.)
- Use double quotes around table and column names that contain mixed case, special characters, or spaces
- Example: SELECT "columnName", "TableName"."ColumnName" FROM "TableName"
- If table/column names are provided in mixed case, preserve their case in the SQL
- For standard identifiers, you can use them without quotes, but be consistent
- Always verify the exact case of table and column names from the schema

Here are some examples:
""" + "\n".join([f"NL: {ex['nl']}\nSQL: {ex['sql']}" for ex in FEW_SHOTS])

REFINEMENT_PROMPT = """
Given a natural language query, suggest 2-3 rephrased or alternative queries that could provide similar or more detailed insights. Focus on clarity, specificity, and variations like time ranges or aggregations.
"""

def generate_query_suggestions(natural_language: str, schema_snapshot: dict = None) -> list[str]:
    schema_info = f"Schema: {schema_snapshot}" if schema_snapshot else ""
    prompt = REFINEMENT_PROMPT + f"\n{schema_info}\nOriginal Query: {natural_language}\nSuggestions:"
    response = openai.ChatCompletion.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[{"role": "user", "content": prompt}]
    )
    suggestions_text = response["choices"][0]["message"]["content"].strip()
    # Parse into list, assuming comma-separated or numbered
    suggestions = [s.strip() for s in suggestions_text.split('\n') if s.strip()]
    return suggestions[:3]  # Limit to 3


def generate_sql(natural_language: str, schema_snapshot: dict, constraints: GuardrailConfig) -> GeneratedSQL:
    prompt = SYSTEM_PROMPT + f"\nSchema: {schema_snapshot}\nNL: {natural_language}\nSQL:"
    response = openai.ChatCompletion.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    )
    sql = response["choices"][0]["message"]["content"].strip()
    safe_sql, added_limit = sanitize_sql(sql, constraints)
    rationale = "Generated with schema priming and guardrails."
    suggestions = generate_query_suggestions(natural_language, schema_snapshot)
    return GeneratedSQL(sql=safe_sql, rationale=rationale, added_limit=added_limit, suggestions=suggestions)
