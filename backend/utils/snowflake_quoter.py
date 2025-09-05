"""
Snowflake SQL Identifier Quoting Utility
Automatically quotes table and column names for Snowflake compatibility
"""

import re
import sqlparse
from typing import List, Set, Dict

class SnowflakeIdentifierQuoter:
    """
    Utility class to automatically quote all identifiers in SQL queries for Snowflake
    """
    
    def __init__(self):
        # Keywords that should not be quoted (SQL reserved words)
        self.sql_keywords = {
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
            'ON', 'AND', 'OR', 'NOT', 'NULL', 'IS', 'IN', 'BETWEEN', 'LIKE',
            'ORDER', 'BY', 'GROUP', 'HAVING', 'LIMIT', 'OFFSET', 'DISTINCT',
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'AS', 'ASC', 'DESC',
            'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP',
            'TABLE', 'COLUMN', 'INDEX', 'VIEW', 'DATABASE', 'SCHEMA',
            'INT', 'INTEGER', 'VARCHAR', 'TEXT', 'DATE', 'TIMESTAMP', 'BOOLEAN',
            'TRUE', 'FALSE', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
            'UNION', 'INTERSECT', 'EXCEPT', 'EXISTS', 'ANY', 'ALL',
            'CURRENT_DATE', 'CURRENT_TIMESTAMP', 'NOW', 'EXTRACT', 'DATE_TRUNC'
        }
        
        # Snowflake-specific functions that should not be quoted
        self.snowflake_functions = {
            'DATEADD', 'DATEDIFF', 'DATE_PART', 'DATE_TRUNC', 'CURRENT_DATE',
            'CURRENT_TIMESTAMP', 'LISTAGG', 'ARRAY_AGG', 'OBJECT_CONSTRUCT',
            'PARSE_JSON', 'TRY_PARSE_JSON', 'FLATTEN', 'LATERAL'
        }
        
        self.all_keywords = self.sql_keywords.union(self.snowflake_functions)
    
    def needs_quoting(self, identifier: str) -> bool:
        """
        Determine if an identifier needs to be quoted for Snowflake
        """
        if not identifier or identifier.upper() in self.all_keywords:
            return False
            
        # Already quoted
        if identifier.startswith('"') and identifier.endswith('"'):
            return False
            
        # Contains numbers, special characters, or mixed case
        if (re.search(r'[0-9_]', identifier) or 
            not identifier.isalpha() or 
            identifier != identifier.upper()):
            return True
            
        return False
    
    def quote_identifier(self, identifier: str) -> str:
        """
        Quote a single identifier if needed
        """
        if self.needs_quoting(identifier):
            return f'"{identifier}"'
        return identifier
    
    def quote_sql_query(self, sql: str) -> str:
        """
        Automatically quote all identifiers in a SQL query for Snowflake
        """
        try:
            # Parse the SQL
            parsed = sqlparse.parse(sql)[0]
            
            # Process tokens and quote identifiers
            result = self._process_tokens(parsed.tokens)
            
            return result
            
        except Exception as e:
            print(f"Warning: Could not parse SQL for identifier quoting: {e}")
            # Fallback: use regex-based approach
            return self._regex_quote_identifiers(sql)
    
    def _process_tokens(self, tokens) -> str:
        """
        Process parsed SQL tokens and quote identifiers
        """
        result = []
        
        for token in tokens:
            if token.ttype is sqlparse.tokens.Name:
                # This is likely a table or column name
                quoted = self.quote_identifier(str(token))
                result.append(quoted)
            elif hasattr(token, 'tokens'):
                # Recursively process sub-tokens
                result.append(self._process_tokens(token.tokens))
            else:
                result.append(str(token))
        
        return ''.join(result)
    
    def _regex_quote_identifiers(self, sql: str) -> str:
        """
        Fallback regex-based identifier quoting
        """
        # Pattern to match potential identifiers (avoiding keywords and already quoted)
        # This is a simple approach - matches words that contain numbers or underscores
        pattern = r'\b([a-zA-Z][a-zA-Z0-9_]*[0-9_][a-zA-Z0-9_]*)\b'
        
        def replace_match(match):
            identifier = match.group(1)
            if identifier.upper() not in self.all_keywords:
                return f'"{identifier}"'
            return identifier
        
        return re.sub(pattern, replace_match, sql)

# Global instance for easy use
snowflake_quoter = SnowflakeIdentifierQuoter()

def quote_snowflake_sql(sql: str) -> str:
    """
    Convenience function to quote SQL identifiers for Snowflake
    """
    return snowflake_quoter.quote_sql_query(sql)

def quote_table_name(table_name: str) -> str:
    """
    Quote a table name for Snowflake if needed
    """
    return snowflake_quoter.quote_identifier(table_name)

def quote_column_name(column_name: str) -> str:
    """
    Quote a column name for Snowflake if needed
    """
    return snowflake_quoter.quote_identifier(column_name)
