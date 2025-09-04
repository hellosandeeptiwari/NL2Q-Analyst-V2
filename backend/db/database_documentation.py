"""
Database-Specific Documentation and Syntax Guide
Provides LLM with database-specific SQL syntax, functions, and capabilities
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class DatabaseDialect:
    """Database-specific syntax and capabilities"""
    name: str
    version: Optional[str]
    sql_dialect: str
    case_sensitivity: str
    quote_char: str
    limit_syntax: str
    date_functions: Dict[str, str]
    string_functions: Dict[str, str]
    aggregate_functions: List[str]
    window_functions: List[str]
    special_features: List[str]
    common_patterns: Dict[str, str]
    performance_tips: List[str]
    limitations: List[str]

# Database-specific configurations
DATABASE_DIALECTS = {
    "postgresql": DatabaseDialect(
        name="PostgreSQL",
        version=None,
        sql_dialect="ANSI SQL with PostgreSQL extensions",
        case_sensitivity="Case-sensitive identifiers, case-insensitive keywords",
        quote_char='"',
        limit_syntax="LIMIT n OFFSET m",
        date_functions={
            "current_date": "CURRENT_DATE",
            "current_timestamp": "CURRENT_TIMESTAMP",
            "date_trunc": "DATE_TRUNC('day', column)",
            "extract": "EXTRACT(YEAR FROM date_column)",
            "age": "AGE(date1, date2)",
            "interval": "INTERVAL '1 month'"
        },
        string_functions={
            "concat": "CONCAT(str1, str2) or str1 || str2",
            "substring": "SUBSTRING(string FROM start FOR length)",
            "upper": "UPPER(column)",
            "lower": "LOWER(column)",
            "trim": "TRIM(column)",
            "replace": "REPLACE(string, from_str, to_str)"
        },
        aggregate_functions=["COUNT", "SUM", "AVG", "MIN", "MAX", "STDDEV", "VARIANCE", "STRING_AGG", "ARRAY_AGG"],
        window_functions=["ROW_NUMBER", "RANK", "DENSE_RANK", "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE"],
        special_features=[
            "JSON/JSONB support", "Arrays", "CTEs", "Window functions", "Full-text search",
            "Recursive queries", "UPSERT (ON CONFLICT)", "Materialized views"
        ],
        common_patterns={
            "pagination": "SELECT * FROM table ORDER BY id LIMIT 10 OFFSET 20",
            "upsert": "INSERT INTO table (id, name) VALUES (1, 'John') ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
            "json_query": "SELECT data->>'name' FROM table WHERE data @> '{\"active\": true}'",
            "date_range": "SELECT * FROM table WHERE date_col BETWEEN '2024-01-01' AND '2024-12-31'"
        },
        performance_tips=[
            "Use EXPLAIN ANALYZE to check query plans",
            "Create indexes on frequently queried columns",
            "Use LIMIT for large result sets",
            "Consider partitioning for very large tables",
            "Use appropriate JOIN types based on data relationships"
        ],
        limitations=[
            "No recursive CTEs in some older versions",
            "JSON functions vary by version",
            "Some window functions require newer versions"
        ]
    ),
    
    "snowflake": DatabaseDialect(
        name="Snowflake",
        version=None,
        sql_dialect="ANSI SQL with Snowflake extensions",
        case_sensitivity="Case-insensitive by default, but preserves case in quotes",
        quote_char='"',
        limit_syntax="LIMIT n",
        date_functions={
            "current_date": "CURRENT_DATE()",
            "current_timestamp": "CURRENT_TIMESTAMP()",
            "date_trunc": "DATE_TRUNC('DAY', column)",
            "extract": "EXTRACT(YEAR FROM date_column)",
            "datediff": "DATEDIFF('day', date1, date2)",
            "dateadd": "DATEADD('month', 1, date_column)"
        },
        string_functions={
            "concat": "CONCAT(str1, str2) or str1 || str2",
            "substring": "SUBSTRING(string, start, length)",
            "upper": "UPPER(column)",
            "lower": "LOWER(column)",
            "trim": "TRIM(column)",
            "replace": "REPLACE(string, old_str, new_str)"
        },
        aggregate_functions=["COUNT", "SUM", "AVG", "MIN", "MAX", "STDDEV", "VARIANCE", "LISTAGG", "ARRAY_AGG"],
        window_functions=["ROW_NUMBER", "RANK", "DENSE_RANK", "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE", "PERCENTILE_CONT"],
        special_features=[
            "Time Travel", "Zero-copy cloning", "Automatic scaling", "Semi-structured data",
            "Secure data sharing", "Streams and tasks", "User-defined functions"
        ],
        common_patterns={
            "time_travel": "SELECT * FROM table AT(TIMESTAMP => '2024-01-01 12:00:00')",
            "pivot": "SELECT * FROM (SELECT...) PIVOT(SUM(amount) FOR category IN ('A', 'B', 'C'))",
            "json_query": "SELECT value:name::string FROM table, LATERAL FLATTEN(input => json_column)",
            "window_function": "SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY date) FROM table"
        },
        performance_tips=[
            "Use clustering keys for large tables",
            "Leverage automatic query optimization",
            "Use LIMIT to control result size",
            "Consider materialized views for complex aggregations",
            "Use appropriate warehouse sizes for workload"
        ],
        limitations=[
            "Time Travel has retention limits",
            "Some functions require specific editions",
            "Cost considerations for large scans"
        ]
    ),
    
    "mysql": DatabaseDialect(
        name="MySQL",
        version=None,
        sql_dialect="MySQL SQL dialect",
        case_sensitivity="Case-insensitive on Windows, case-sensitive on Unix",
        quote_char="`",
        limit_syntax="LIMIT n OFFSET m",
        date_functions={
            "current_date": "CURDATE()",
            "current_timestamp": "NOW()",
            "date_format": "DATE_FORMAT(date, '%Y-%m-%d')",
            "extract": "EXTRACT(YEAR FROM date_column)",
            "datediff": "DATEDIFF(date1, date2)",
            "date_add": "DATE_ADD(date, INTERVAL 1 MONTH)"
        },
        string_functions={
            "concat": "CONCAT(str1, str2)",
            "substring": "SUBSTRING(string, start, length)",
            "upper": "UPPER(column)",
            "lower": "LOWER(column)",
            "trim": "TRIM(column)",
            "replace": "REPLACE(string, old_str, new_str)"
        },
        aggregate_functions=["COUNT", "SUM", "AVG", "MIN", "MAX", "STD", "VARIANCE", "GROUP_CONCAT"],
        window_functions=["ROW_NUMBER", "RANK", "DENSE_RANK", "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE"],
        special_features=[
            "Full-text indexing", "JSON support (5.7+)", "Generated columns",
            "Partitioning", "Replication", "Storage engines"
        ],
        common_patterns={
            "pagination": "SELECT * FROM table ORDER BY id LIMIT 10 OFFSET 20",
            "upsert": "INSERT INTO table (id, name) VALUES (1, 'John') ON DUPLICATE KEY UPDATE name = VALUES(name)",
            "full_text": "SELECT * FROM table WHERE MATCH(content) AGAINST('search term' IN BOOLEAN MODE)",
            "json_query": "SELECT JSON_EXTRACT(data, '$.name') FROM table WHERE JSON_EXTRACT(data, '$.active') = true"
        },
        performance_tips=[
            "Use EXPLAIN to analyze queries",
            "Create appropriate indexes",
            "Use LIMIT for large datasets",
            "Choose right storage engine (InnoDB recommended)",
            "Optimize JOIN order"
        ],
        limitations=[
            "Window functions only in MySQL 8.0+",
            "Limited CTE support",
            "JSON functions vary by version"
        ]
    ),
    
    "sqlite": DatabaseDialect(
        name="SQLite",
        version=None,
        sql_dialect="SQLite SQL dialect",
        case_sensitivity="Case-insensitive by default",
        quote_char='"',
        limit_syntax="LIMIT n OFFSET m",
        date_functions={
            "current_date": "date('now')",
            "current_timestamp": "datetime('now')",
            "date": "date(column)",
            "strftime": "strftime('%Y-%m-%d', column)",
            "julianday": "julianday(date1) - julianday(date2)"
        },
        string_functions={
            "concat": "str1 || str2",
            "substring": "substr(string, start, length)",
            "upper": "upper(column)",
            "lower": "lower(column)",
            "trim": "trim(column)",
            "replace": "replace(string, old_str, new_str)"
        },
        aggregate_functions=["COUNT", "SUM", "AVG", "MIN", "MAX", "GROUP_CONCAT", "TOTAL"],
        window_functions=["ROW_NUMBER", "RANK", "DENSE_RANK", "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE"],
        special_features=[
            "Lightweight and embedded", "Full-text search with FTS", "JSON1 extension",
            "R-Tree spatial indexing", "Virtual tables", "Triggers"
        ],
        common_patterns={
            "pagination": "SELECT * FROM table ORDER BY id LIMIT 10 OFFSET 20",
            "upsert": "INSERT OR REPLACE INTO table (id, name) VALUES (1, 'John')",
            "date_calc": "SELECT * FROM table WHERE date(created_at) = date('now', '-1 day')",
            "json_query": "SELECT json_extract(data, '$.name') FROM table WHERE json_extract(data, '$.active') = 1"
        },
        performance_tips=[
            "Create indexes on frequently queried columns",
            "Use ANALYZE to update statistics",
            "Consider VACUUM for database maintenance",
            "Use prepared statements to avoid SQL injection",
            "Limit result sets with LIMIT clause"
        ],
        limitations=[
            "No stored procedures",
            "Limited ALTER TABLE support",
            "No right/full outer joins",
            "Single writer at a time"
        ]
    )
}

def get_database_documentation(engine_type: str, version: Optional[str] = None) -> DatabaseDialect:
    """Get database-specific documentation and syntax guide"""
    
    # Normalize engine type
    engine_lower = engine_type.lower()
    
    # Map common variations
    if engine_lower in ['postgres', 'postgresql', 'psql']:
        engine_lower = 'postgresql'
    elif engine_lower in ['snow', 'snowflake']:
        engine_lower = 'snowflake'
    elif engine_lower in ['mysql', 'mariadb']:
        engine_lower = 'mysql'
    elif engine_lower in ['sqlite', 'sqlite3']:
        engine_lower = 'sqlite'
    
    # Return specific dialect or default
    if engine_lower in DATABASE_DIALECTS:
        dialect = DATABASE_DIALECTS[engine_lower]
        if version:
            dialect.version = version
        return dialect
    
    # Fallback for unknown databases
    return DatabaseDialect(
        name=engine_type,
        version=version,
        sql_dialect="Standard SQL",
        case_sensitivity="Database-dependent",
        quote_char='"',
        limit_syntax="LIMIT n",
        date_functions={"current_date": "CURRENT_DATE", "current_timestamp": "CURRENT_TIMESTAMP"},
        string_functions={"concat": "CONCAT(str1, str2)", "upper": "UPPER(column)", "lower": "LOWER(column)"},
        aggregate_functions=["COUNT", "SUM", "AVG", "MIN", "MAX"],
        window_functions=["ROW_NUMBER", "RANK", "DENSE_RANK"],
        special_features=["Standard SQL compliance"],
        common_patterns={"basic_select": "SELECT * FROM table WHERE condition"},
        performance_tips=["Use indexes", "Limit result sets", "Optimize WHERE clauses"],
        limitations=["Unknown database - using standard SQL assumptions"]
    )

def format_database_documentation_for_llm(dialect: DatabaseDialect) -> str:
    """Format database documentation for LLM consumption"""
    
    doc = f"""
DATABASE-SPECIFIC SYNTAX GUIDE: {dialect.name}
{'=' * 50}

ENGINE DETAILS:
- Database: {dialect.name} {dialect.version or ''}
- SQL Dialect: {dialect.sql_dialect}
- Case Sensitivity: {dialect.case_sensitivity}
- Quote Character: {dialect.quote_char}
- Limit Syntax: {dialect.limit_syntax}

DATE/TIME FUNCTIONS:
"""
    
    for func, syntax in dialect.date_functions.items():
        doc += f"- {func.upper()}: {syntax}\n"
    
    doc += "\nSTRING FUNCTIONS:\n"
    for func, syntax in dialect.string_functions.items():
        doc += f"- {func.upper()}: {syntax}\n"
    
    doc += f"\nAGGREGATE FUNCTIONS:\n"
    doc += f"Available: {', '.join(dialect.aggregate_functions)}\n"
    
    doc += f"\nWINDOW FUNCTIONS:\n"
    doc += f"Available: {', '.join(dialect.window_functions)}\n"
    
    doc += f"\nSPECIAL FEATURES:\n"
    for feature in dialect.special_features:
        doc += f"- {feature}\n"
    
    doc += f"\nCOMMON PATTERNS:\n"
    for pattern, example in dialect.common_patterns.items():
        doc += f"- {pattern.upper()}: {example}\n"
    
    doc += f"\nPERFORMANCE TIPS:\n"
    for tip in dialect.performance_tips:
        doc += f"- {tip}\n"
    
    if dialect.limitations:
        doc += f"\nLIMITATIONS TO AVOID:\n"
        for limitation in dialect.limitations:
            doc += f"- {limitation}\n"
    
    doc += f"\nSYNTAX REQUIREMENTS:\n"
    doc += f"- Use {dialect.quote_char} for identifiers with spaces or special characters\n"
    doc += f"- Use {dialect.limit_syntax} for result limiting\n"
    doc += f"- Follow {dialect.sql_dialect} standards\n"
    
    return doc.strip()

def get_database_specific_examples(engine_type: str) -> List[Dict[str, str]]:
    """Get database-specific query examples"""
    
    engine_lower = engine_type.lower()
    
    if engine_lower in ['postgresql', 'postgres']:
        return [
            {
                "description": "PostgreSQL date range with timezone",
                "sql": "SELECT * FROM events WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'"
            },
            {
                "description": "PostgreSQL JSON query",
                "sql": "SELECT id, data->>'name' as name FROM users WHERE data @> '{\"active\": true}'"
            },
            {
                "description": "PostgreSQL window function",
                "sql": "SELECT *, ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) FROM employees"
            }
        ]
    
    elif engine_lower == 'snowflake':
        return [
            {
                "description": "Snowflake date arithmetic",
                "sql": "SELECT * FROM table WHERE date_col >= DATEADD('day', -7, CURRENT_DATE())"
            },
            {
                "description": "Snowflake PIVOT operation",
                "sql": "SELECT * FROM (SELECT product, month, sales FROM sales_data) PIVOT(SUM(sales) FOR month IN ('Jan', 'Feb', 'Mar'))"
            },
            {
                "description": "Snowflake flatten JSON",
                "sql": "SELECT f.value:name::string as name FROM table, LATERAL FLATTEN(input => json_column) f"
            }
        ]
    
    elif engine_lower == 'mysql':
        return [
            {
                "description": "MySQL date functions",
                "sql": "SELECT * FROM orders WHERE DATE(created_at) = CURDATE()"
            },
            {
                "description": "MySQL GROUP_CONCAT",
                "sql": "SELECT category, GROUP_CONCAT(product_name SEPARATOR ', ') FROM products GROUP BY category"
            },
            {
                "description": "MySQL UPSERT",
                "sql": "INSERT INTO stats (id, count) VALUES (1, 1) ON DUPLICATE KEY UPDATE count = count + 1"
            }
        ]
    
    elif engine_lower == 'sqlite':
        return [
            {
                "description": "SQLite date calculations",
                "sql": "SELECT * FROM logs WHERE date(timestamp) = date('now', '-1 day')"
            },
            {
                "description": "SQLite string concatenation",
                "sql": "SELECT first_name || ' ' || last_name as full_name FROM users"
            },
            {
                "description": "SQLite CASE expression",
                "sql": "SELECT *, CASE WHEN age < 18 THEN 'Minor' ELSE 'Adult' END as category FROM users"
            }
        ]
    
    return [
        {
            "description": "Standard SQL query",
            "sql": "SELECT COUNT(*) FROM table WHERE condition = 'value'"
        }
    ]
