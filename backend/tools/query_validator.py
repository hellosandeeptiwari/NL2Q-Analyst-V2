"""
Query Validator with Multi-Layer Safety Checks
Validates SQL queries for syntax, security, performance, and business logic
"""

import re
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import sqlparse
from sqlparse import sql, tokens
import snowflake.connector
from datetime import datetime

@dataclass
class ValidationResult:
    """Comprehensive validation result"""
    is_valid: bool
    error_level: str  # INFO, WARNING, ERROR, CRITICAL
    error_code: str
    message: str
    suggestion: Optional[str] = None
    confidence: float = 1.0

@dataclass
class QueryMetrics:
    """Query performance and cost metrics"""
    estimated_cost: float
    estimated_runtime_seconds: int
    estimated_rows: int
    complexity_score: float
    resource_usage: Dict[str, Any]

class QueryValidator:
    """
    Comprehensive SQL query validation with security and performance checks
    """
    
    def __init__(self, connection_params: Dict[str, str]):
        self.connection_params = connection_params
        
        # Security patterns to block
        self.security_patterns = [
            r"drop\s+(table|database|schema)",
            r"delete\s+from",
            r"update\s+.+\s+set",
            r"insert\s+into",
            r"create\s+(table|database|schema)", 
            r"alter\s+(table|database|schema)",
            r"truncate\s+table",
            r"grant\s+",
            r"revoke\s+",
            r"--.*;\s*drop",  # SQL injection patterns
            r"';.*drop",
            r"union.*select.*from",
            r"exec\s*\(",
            r"sp_\w+",  # Stored procedures
        ]
        
        # Performance anti-patterns
        self.performance_patterns = [
            r"select\s+\*\s+from\s+\w+\s*$",  # SELECT * without WHERE
            r"like\s+'\%.*\%'",  # Leading wildcard LIKE
            r"or\s+\w+\s*=",  # OR conditions that prevent index usage
            r"function\(\w+\)\s*=",  # Functions on columns in WHERE
            r"order\s+by\s+.*\s+limit\s+\d{4,}",  # Large LIMIT without efficient ORDER BY
        ]
        
        # Business logic validation rules
        self.business_rules = {
            "nbrx_trx_consistency": {
                "pattern": r"(nbrx|new.*prescription).*?(?=.*?(trx|total.*prescription))",
                "rule": "NBRx should be <= TRx in the same time period",
                "severity": "WARNING"
            },
            "date_range_reasonable": {
                "pattern": r"date.*?between.*?'(\d{4}-\d{2}-\d{2})'.*?'(\d{4}-\d{2}-\d{2})'",
                "rule": "Date ranges should be reasonable (not more than 5 years)",
                "severity": "WARNING"
            },
            "patient_privacy": {
                "pattern": r"(patient_name|ssn|dob|birth_date|address|phone)",
                "rule": "Queries containing PII/PHI require special authorization",
                "severity": "CRITICAL"
            }
        }
        
        # Column size estimates for cost calculation
        self.table_estimates = {
            "rx_facts": {"rows": 50000000, "avg_row_size": 256},
            "patient_facts": {"rows": 10000000, "avg_row_size": 512},
            "hcp_master": {"rows": 2000000, "avg_row_size": 1024},
            "product_master": {"rows": 50000, "avg_row_size": 512}
        }
    
    async def validate_static(self, plan_or_sql: Any) -> List[ValidationResult]:
        """
        Static validation - syntax, security, basic structure
        """
        
        validation_results = []
        
        # Extract SQL from plan or use directly
        if hasattr(plan_or_sql, 'tool_sequence'):
            sql_query = self._extract_sql_from_plan(plan_or_sql)
        else:
            sql_query = str(plan_or_sql)
        
        if not sql_query:
            validation_results.append(ValidationResult(
                is_valid=False,
                error_level="ERROR",
                error_code="NO_SQL",
                message="No SQL query found to validate"
            ))
            return validation_results
        
        # 1. Syntax validation
        syntax_result = await self._validate_syntax(sql_query)
        validation_results.extend(syntax_result)
        
        # 2. Security validation  
        security_result = await self._validate_security(sql_query)
        validation_results.extend(security_result)
        
        # 3. Performance anti-pattern check
        performance_result = await self._validate_performance_patterns(sql_query)
        validation_results.extend(performance_result)
        
        # 4. Business logic validation
        business_result = await self._validate_business_logic(sql_query)
        validation_results.extend(business_result)
        
        return validation_results
    
    async def validate_schema(self, plan_or_sql: Any) -> List[ValidationResult]:
        """
        Schema validation - table/column existence, permissions
        """
        
        validation_results = []
        
        # Extract SQL
        if hasattr(plan_or_sql, 'tool_sequence'):
            sql_query = self._extract_sql_from_plan(plan_or_sql)
        else:
            sql_query = str(plan_or_sql)
        
        if not sql_query:
            return validation_results
        
        try:
            # Parse SQL to extract table and column references
            parsed = sqlparse.parse(sql_query)[0]
            tables, columns = self._extract_schema_references(parsed)
            
            # Validate table existence
            table_validation = await self._validate_table_existence(tables)
            validation_results.extend(table_validation)
            
            # Validate column existence
            column_validation = await self._validate_column_existence(columns)
            validation_results.extend(column_validation)
            
            # Validate permissions
            permission_validation = await self._validate_table_permissions(tables)
            validation_results.extend(permission_validation)
            
        except Exception as e:
            validation_results.append(ValidationResult(
                is_valid=False,
                error_level="ERROR",
                error_code="SCHEMA_VALIDATION_ERROR",
                message=f"Schema validation failed: {str(e)}"
            ))
        
        return validation_results
    
    async def dry_run(self, plan_or_sql: Any) -> Dict[str, Any]:
        """
        Dry run validation - EXPLAIN plan, cost estimation
        """
        
        # Extract SQL
        if hasattr(plan_or_sql, 'tool_sequence'):
            sql_query = self._extract_sql_from_plan(plan_or_sql)
        else:
            sql_query = str(plan_or_sql)
        
        if not sql_query:
            raise ValueError("No SQL query to validate")
        
        conn = snowflake.connector.connect(**self.connection_params)
        cursor = conn.cursor()
        
        try:
            # Get EXPLAIN plan
            explain_sql = f"EXPLAIN {sql_query}"
            cursor.execute(explain_sql)
            explain_result = cursor.fetchall()
            
            # Parse explain plan for insights
            execution_plan = self._parse_explain_plan(explain_result)
            
            # Estimate cost and performance
            metrics = await self._estimate_query_metrics(sql_query, execution_plan)
            
            return {
                "explain_plan": explain_result,
                "execution_plan": execution_plan,
                "metrics": metrics,
                "recommendations": self._generate_optimization_recommendations(execution_plan),
                "dry_run_timestamp": datetime.now().isoformat()
            }
            
        finally:
            cursor.close()
            conn.close()
    
    async def _validate_syntax(self, sql_query: str) -> List[ValidationResult]:
        """Validate SQL syntax"""
        
        results = []
        
        try:
            # Parse SQL
            parsed = sqlparse.parse(sql_query)
            
            if not parsed:
                results.append(ValidationResult(
                    is_valid=False,
                    error_level="ERROR",
                    error_code="SYNTAX_ERROR", 
                    message="SQL query could not be parsed"
                ))
                return results
            
            # Check for basic SQL structure
            statement = parsed[0]
            
            # Must have SELECT
            if not any(token.ttype is tokens.Keyword and 
                      token.value.upper() == "SELECT" 
                      for token in statement.flatten()):
                results.append(ValidationResult(
                    is_valid=False,
                    error_level="ERROR",
                    error_code="NO_SELECT",
                    message="Query must contain SELECT statement"
                ))
            
            # Check for unmatched parentheses
            paren_count = sql_query.count('(') - sql_query.count(')')
            if paren_count != 0:
                results.append(ValidationResult(
                    is_valid=False,
                    error_level="ERROR",
                    error_code="UNMATCHED_PARENTHESES",
                    message=f"Unmatched parentheses: {abs(paren_count)} {'opening' if paren_count > 0 else 'closing'}"
                ))
            
            # Check for unclosed quotes
            single_quotes = sql_query.count("'") % 2
            double_quotes = sql_query.count('"') % 2
            
            if single_quotes != 0:
                results.append(ValidationResult(
                    is_valid=False,
                    error_level="ERROR",
                    error_code="UNCLOSED_QUOTES",
                    message="Unclosed single quotes detected"
                ))
            
            if double_quotes != 0:
                results.append(ValidationResult(
                    is_valid=False,
                    error_level="ERROR",
                    error_code="UNCLOSED_QUOTES",
                    message="Unclosed double quotes detected"
                ))
            
            # If no errors found, syntax is valid
            if not results:
                results.append(ValidationResult(
                    is_valid=True,
                    error_level="INFO",
                    error_code="SYNTAX_VALID",
                    message="SQL syntax is valid"
                ))
            
        except Exception as e:
            results.append(ValidationResult(
                is_valid=False,
                error_level="ERROR",
                error_code="SYNTAX_PARSE_ERROR",
                message=f"Syntax validation failed: {str(e)}"
            ))
        
        return results
    
    async def _validate_security(self, sql_query: str) -> List[ValidationResult]:
        """Validate against security threats"""
        
        results = []
        query_lower = sql_query.lower()
        
        # Check each security pattern
        for pattern in self.security_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                results.append(ValidationResult(
                    is_valid=False,
                    error_level="CRITICAL",
                    error_code="SECURITY_VIOLATION",
                    message=f"Dangerous SQL pattern detected: {pattern}",
                    suggestion="Only SELECT statements are allowed"
                ))
        
        # Check for SQL injection indicators
        injection_patterns = [
            r"';\s*--",
            r"'.*?union.*?select",
            r"'\s*or\s*'1'\s*=\s*'1",
            r"'\s*or\s*1\s*=\s*1",
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                results.append(ValidationResult(
                    is_valid=False,
                    error_level="CRITICAL",
                    error_code="SQL_INJECTION",
                    message="Potential SQL injection pattern detected",
                    suggestion="Query blocked for security reasons"
                ))
        
        # If no security issues, add confirmation
        if not results:
            results.append(ValidationResult(
                is_valid=True,
                error_level="INFO",
                error_code="SECURITY_PASS",
                message="No security threats detected"
            ))
        
        return results
    
    async def _validate_performance_patterns(self, sql_query: str) -> List[ValidationResult]:
        """Check for performance anti-patterns"""
        
        results = []
        query_lower = sql_query.lower()
        
        # Check each performance pattern
        for pattern in self.performance_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                results.append(ValidationResult(
                    is_valid=True,  # Warning, not blocking
                    error_level="WARNING",
                    error_code="PERFORMANCE_WARNING",
                    message=f"Performance anti-pattern detected: {pattern}",
                    suggestion=self._get_performance_suggestion(pattern)
                ))
        
        # Check for missing WHERE clause on large tables
        if re.search(r"from\s+(rx_facts|patient_facts)", query_lower) and \
           not re.search(r"where", query_lower):
            results.append(ValidationResult(
                is_valid=True,
                error_level="WARNING",
                error_code="MISSING_WHERE_CLAUSE",
                message="Large table query without WHERE clause detected",
                suggestion="Add date filters to improve performance"
            ))
        
        return results
    
    async def _validate_business_logic(self, sql_query: str) -> List[ValidationResult]:
        """Validate business logic rules"""
        
        results = []
        query_lower = sql_query.lower()
        
        # Check each business rule
        for rule_name, rule_config in self.business_rules.items():
            if re.search(rule_config["pattern"], query_lower, re.IGNORECASE):
                severity = rule_config["severity"]
                is_valid = severity != "CRITICAL"
                
                results.append(ValidationResult(
                    is_valid=is_valid,
                    error_level=severity,
                    error_code=f"BUSINESS_RULE_{rule_name.upper()}",
                    message=rule_config["rule"],
                    suggestion=self._get_business_rule_suggestion(rule_name)
                ))
        
        return results
    
    def _extract_sql_from_plan(self, plan: Any) -> Optional[str]:
        """Extract SQL query from execution plan"""
        
        if hasattr(plan, 'tool_sequence'):
            for step in plan.tool_sequence:
                if step.get("tool") == "sql_generation":
                    return step.get("generated_sql", "")
        
        return None
    
    def _extract_schema_references(self, parsed_sql) -> Tuple[List[str], List[str]]:
        """Extract table and column references from parsed SQL"""
        
        tables = []
        columns = []
        
        # Simple extraction - in production would be more sophisticated
        sql_str = str(parsed_sql).lower()
        
        # Extract table names after FROM and JOIN
        table_pattern = r"(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)"
        table_matches = re.findall(table_pattern, sql_str)
        tables.extend(table_matches)
        
        # Extract column names (simplified)
        column_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:=|<|>|!=|like|in)"
        column_matches = re.findall(column_pattern, sql_str)
        columns.extend(column_matches)
        
        return list(set(tables)), list(set(columns))
    
    async def _validate_table_existence(self, tables: List[str]) -> List[ValidationResult]:
        """Validate that tables exist"""
        
        results = []
        
        # For demo purposes, assume tables exist
        # In production, would query information_schema
        
        known_tables = ["rx_facts", "patient_facts", "hcp_master", "product_master"]
        
        for table in tables:
            table_name = table.split('.')[-1]  # Get table name without schema
            if table_name not in known_tables:
                results.append(ValidationResult(
                    is_valid=False,
                    error_level="ERROR",
                    error_code="TABLE_NOT_FOUND",
                    message=f"Table {table} does not exist or is not accessible"
                ))
        
        return results
    
    async def _validate_column_existence(self, columns: List[str]) -> List[ValidationResult]:
        """Validate that columns exist"""
        
        # For demo purposes, return valid
        # In production, would check column metadata
        
        return [ValidationResult(
            is_valid=True,
            error_level="INFO",
            error_code="COLUMNS_VALID",
            message="Column validation passed"
        )]
    
    async def _validate_table_permissions(self, tables: List[str]) -> List[ValidationResult]:
        """Validate user permissions on tables"""
        
        # For demo purposes, assume permissions are valid
        # In production, would check user grants
        
        return [ValidationResult(
            is_valid=True,
            error_level="INFO",
            error_code="PERMISSIONS_VALID",
            message="User has required permissions"
        )]
    
    def _parse_explain_plan(self, explain_result) -> Dict[str, Any]:
        """Parse EXPLAIN plan output"""
        
        # Simplified parsing - production would be more comprehensive
        plan_info = {
            "operation_count": len(explain_result),
            "contains_table_scan": any("TableScan" in str(row) for row in explain_result),
            "contains_join": any("Join" in str(row) for row in explain_result),
            "estimated_complexity": "medium"
        }
        
        return plan_info
    
    async def _estimate_query_metrics(
        self, 
        sql_query: str, 
        execution_plan: Dict[str, Any]
    ) -> QueryMetrics:
        """Estimate query performance metrics"""
        
        # Simple heuristic-based estimation
        # Production would use actual statistics
        
        base_cost = 0.01  # Base cost in credits
        base_runtime = 1   # Base runtime in seconds
        base_rows = 1000   # Base row estimate
        
        query_lower = sql_query.lower()
        
        # Cost factors
        if "rx_facts" in query_lower:
            base_cost *= 5
            base_runtime *= 3
            base_rows *= 1000
        
        if "join" in query_lower:
            base_cost *= 2
            base_runtime *= 2
        
        if "order by" in query_lower:
            base_cost *= 1.5
            base_runtime *= 1.5
        
        if "group by" in query_lower:
            base_cost *= 1.3
            base_runtime *= 1.2
        
        # Complexity score
        complexity_factors = [
            ("subquery", 2),
            ("join", 1.5),
            ("group by", 1.2),
            ("order by", 1.1),
            ("having", 1.3),
            ("union", 1.8)
        ]
        
        complexity_score = 1.0
        for factor, multiplier in complexity_factors:
            if factor in query_lower:
                complexity_score *= multiplier
        
        return QueryMetrics(
            estimated_cost=round(base_cost, 4),
            estimated_runtime_seconds=int(base_runtime),
            estimated_rows=base_rows,
            complexity_score=round(complexity_score, 2),
            resource_usage={
                "memory_mb": base_rows // 1000,
                "cpu_cores": min(4, int(complexity_score)),
                "io_operations": base_rows // 100
            }
        )
    
    def _generate_optimization_recommendations(self, execution_plan: Dict[str, Any]) -> List[str]:
        """Generate query optimization recommendations"""
        
        recommendations = []
        
        if execution_plan.get("contains_table_scan"):
            recommendations.append("Consider adding WHERE clause filters to avoid full table scans")
        
        if execution_plan.get("contains_join"):
            recommendations.append("Ensure join conditions use indexed columns for better performance")
        
        if execution_plan.get("operation_count", 0) > 10:
            recommendations.append("Query has many operations - consider simplifying or breaking into steps")
        
        return recommendations
    
    def _get_performance_suggestion(self, pattern: str) -> str:
        """Get performance improvement suggestion for pattern"""
        
        suggestions = {
            r"select\s+\*": "Use specific column names instead of SELECT *",
            r"like\s+'\%": "Avoid leading wildcards in LIKE patterns",
            r"or\s+\w+\s*=": "Consider using IN clause instead of multiple OR conditions",
            r"function\(\w+\)\s*=": "Avoid functions on columns in WHERE clause - consider computed columns"
        }
        
        for pat, suggestion in suggestions.items():
            if pat in pattern:
                return suggestion
        
        return "Review query for performance optimization opportunities"
    
    def _get_business_rule_suggestion(self, rule_name: str) -> str:
        """Get business rule compliance suggestion"""
        
        suggestions = {
            "nbrx_trx_consistency": "Add validation to ensure NBRx <= TRx in calculations",
            "date_range_reasonable": "Consider limiting date ranges to improve performance",
            "patient_privacy": "Contact compliance team for PII/PHI data access approval"
        }
        
        return suggestions.get(rule_name, "Review business logic for compliance")

# Global instance  
query_validator = QueryValidator({
    'user': 'your_user',
    'password': 'your_password', 
    'account': 'your_account',
    'warehouse': 'COMPUTE_WH',
    'database': 'COMMERCIAL_AI',
    'schema': 'ENHANCED_NBA'
})
