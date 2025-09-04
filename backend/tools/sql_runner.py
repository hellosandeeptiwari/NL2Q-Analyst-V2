"""
SQL Runner - Safe execution engine with governance and validation
Implements validation, sandboxing, cost controls, and audit trails
"""

import asyncio
import json
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import sqlparse
from sqlparse import sql, tokens
import hashlib

from backend.db.engine import get_adapter
from backend.governance.rbac_manager import RBACManager
from backend.audit.audit_logger import AuditLogger

@dataclass
class QueryValidationResult:
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = None
    estimated_rows: Optional[int] = None
    estimated_cost: float = 0.0
    execution_plan: Optional[Dict[str, Any]] = None
    safety_score: float = 1.0  # 0-1, higher is safer

@dataclass
class QueryExecutionResult:
    success: bool
    data: List[Dict[str, Any]] = None
    columns: List[str] = None
    row_count: int = 0
    execution_time: float = 0.0
    cost: float = 0.0
    job_id: Optional[str] = None
    error_message: Optional[str] = None
    warnings: List[str] = None
    was_sampled: bool = False
    cache_key: Optional[str] = None

class SQLRunner:
    """
    Enterprise-grade SQL execution engine with comprehensive safety controls
    """
    
    def __init__(self):
        self.db_adapter = get_adapter()
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        
        # Safety configuration
        self.max_execution_time = 30  # seconds
        self.max_rows_default = 10000
        self.max_cost_default = 100.0  # dollars
        self.sample_threshold = 100000  # rows
        
        # Dangerous patterns to block
        self.blocked_patterns = [
            r'\bDROP\s+TABLE\b',
            r'\bDELETE\s+FROM\b',
            r'\bUPDATE\s+.*\s+SET\b',
            r'\bTRUNCATE\s+TABLE\b',
            r'\bALTER\s+TABLE\b',
            r'\bCREATE\s+TABLE\b',
            r'\bINSERT\s+INTO\b',
            r'\bEXEC\s+',
            r'\bEXECUTE\s+',
            r';\s*DROP\s+',  # SQL injection
            r'--\s*DROP\s+',  # Comment injection
        ]
        
        # Allowed functions whitelist
        self.allowed_functions = [
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'ROUND', 'FLOOR', 'CEIL',
            'UPPER', 'LOWER', 'TRIM', 'LENGTH', 'SUBSTRING', 'CONCAT',
            'DATE_TRUNC', 'DATE_PART', 'EXTRACT', 'NOW', 'CURRENT_DATE',
            'COALESCE', 'NULLIF', 'CASE', 'CAST', 'ROW_NUMBER', 'RANK',
            'DENSE_RANK', 'LAG', 'LEAD', 'FIRST_VALUE', 'LAST_VALUE'
        ]
        
    async def validate_query(self, sql: str, schema_context: Dict[str, Any]) -> QueryValidationResult:
        """
        Comprehensive query validation before execution
        """
        
        try:
            # Step 1: Basic safety checks
            safety_result = await self._check_query_safety(sql)
            if not safety_result["is_safe"]:
                return QueryValidationResult(
                    is_valid=False,
                    error_message=safety_result["error"],
                    safety_score=0.0
                )
            
            # Step 2: SQL syntax validation
            syntax_result = await self._validate_sql_syntax(sql)
            if not syntax_result["is_valid"]:
                return QueryValidationResult(
                    is_valid=False,
                    error_message=syntax_result["error"]
                )
            
            # Step 3: Schema validation
            schema_result = await self._validate_schema_references(sql, schema_context)
            if not schema_result["is_valid"]:
                return QueryValidationResult(
                    is_valid=False,
                    error_message=schema_result["error"]
                )
            
            # Step 4: Performance analysis
            performance_result = await self._analyze_query_performance(sql)
            
            # Step 5: Cost estimation
            cost_estimate = await self._estimate_query_cost(sql, schema_context)
            
            # Step 6: Generate execution plan (dry run)
            execution_plan = await self._get_execution_plan(sql)
            
            warnings = []
            warnings.extend(safety_result.get("warnings", []))
            warnings.extend(performance_result.get("warnings", []))
            
            return QueryValidationResult(
                is_valid=True,
                warnings=warnings,
                estimated_rows=performance_result.get("estimated_rows"),
                estimated_cost=cost_estimate,
                execution_plan=execution_plan,
                safety_score=safety_result["safety_score"]
            )
            
        except Exception as e:
            return QueryValidationResult(
                is_valid=False,
                error_message=f"Validation error: {str(e)}"
            )
    
    async def execute_query(
        self, 
        sql: str, 
        user_id: str,
        timeout_seconds: Optional[int] = None,
        max_rows: Optional[int] = None,
        force_sample: bool = False
    ) -> QueryExecutionResult:
        """
        Execute SQL query with comprehensive safety controls
        """
        
        start_time = time.time()
        timeout = timeout_seconds or self.max_execution_time
        row_limit = max_rows or self.max_rows_default
        
        # Generate job ID for tracking
        job_id = self._generate_job_id(sql, user_id)
        
        try:
            # Step 1: Permission check
            permission_check = await self.rbac_manager.check_query_permissions(
                user_id=user_id,
                sql=sql
            )
            
            if not permission_check.get("allowed", False):
                return QueryExecutionResult(
                    success=False,
                    error_message=permission_check.get("error", "Permission denied"),
                    job_id=job_id
                )
            
            # Step 2: Determine if sampling is needed
            should_sample = force_sample or await self._should_sample_query(sql)
            
            if should_sample:
                sql = await self._add_sampling_clause(sql, self.sample_threshold)
                was_sampled = True
            else:
                was_sampled = False
            
            # Step 3: Add row limit for safety
            sql = await self._add_row_limit(sql, row_limit)
            
            # Step 4: Execute with timeout
            execution_result = await self._execute_with_timeout(sql, timeout, job_id)
            
            execution_time = time.time() - start_time
            
            # Step 5: Process results
            if execution_result["success"]:
                data = execution_result["data"]
                columns = execution_result["columns"]
                
                # Apply any necessary data masking
                masked_data = await self._apply_data_masking(data, user_id)
                
                result = QueryExecutionResult(
                    success=True,
                    data=masked_data,
                    columns=columns,
                    row_count=len(masked_data),
                    execution_time=execution_time,
                    cost=execution_result.get("cost", 0.0),
                    job_id=job_id,
                    was_sampled=was_sampled
                )
                
                # Log successful execution
                await self.audit_logger.log_query_execution(
                    user_id=user_id,
                    sql=sql,
                    job_id=job_id,
                    row_count=result.row_count,
                    execution_time=execution_time,
                    cost=result.cost
                )
                
                return result
            
            else:
                return QueryExecutionResult(
                    success=False,
                    error_message=execution_result["error"],
                    execution_time=execution_time,
                    job_id=job_id
                )
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            # Log failed execution
            await self.audit_logger.log_query_error(
                user_id=user_id,
                sql=sql,
                job_id=job_id,
                error=error_msg,
                execution_time=execution_time
            )
            
            return QueryExecutionResult(
                success=False,
                error_message=error_msg,
                execution_time=execution_time,
                job_id=job_id
            )
    
    async def _check_query_safety(self, sql: str) -> Dict[str, Any]:
        """Check query for dangerous patterns and safety issues"""
        
        sql_upper = sql.upper()
        warnings = []
        safety_score = 1.0
        
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, sql_upper):
                return {
                    "is_safe": False,
                    "error": f"Query contains blocked pattern: {pattern}",
                    "safety_score": 0.0
                }
        
        # Check for risky patterns that generate warnings
        risky_patterns = [
            (r'\bSELECT\s+\*\s+FROM\b', "SELECT * can be expensive on large tables"),
            (r'\bCROSS\s+JOIN\b', "CROSS JOIN can produce very large result sets"),
            (r'\bUNION\s+(?!ALL)\b', "UNION without ALL requires deduplication which can be slow"),
            (r'WHERE\s+.*\s+LIKE\s+\'%.*%\'', "Leading wildcard LIKE queries are slow"),
        ]
        
        for pattern, warning in risky_patterns:
            if re.search(pattern, sql_upper):
                warnings.append(warning)
                safety_score -= 0.1
        
        # Check function whitelist
        functions_used = re.findall(r'\b([A-Z_]+)\s*\(', sql_upper)
        for func in functions_used:
            if func not in self.allowed_functions and func not in ['SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'HAVING']:
                warnings.append(f"Function {func} is not in allowed list")
                safety_score -= 0.05
        
        return {
            "is_safe": True,
            "warnings": warnings,
            "safety_score": max(safety_score, 0.0)
        }
    
    async def _validate_sql_syntax(self, sql: str) -> Dict[str, Any]:
        """Validate SQL syntax using sqlparse"""
        
        try:
            parsed = sqlparse.parse(sql)
            
            if not parsed:
                return {"is_valid": False, "error": "Empty or invalid SQL"}
            
            # Check for multiple statements (could be injection)
            if len(parsed) > 1:
                return {"is_valid": False, "error": "Multiple SQL statements not allowed"}
            
            # Basic structure validation
            statement = parsed[0]
            if statement.get_type() != 'SELECT':
                return {"is_valid": False, "error": "Only SELECT statements are allowed"}
            
            return {"is_valid": True}
            
        except Exception as e:
            return {"is_valid": False, "error": f"SQL syntax error: {str(e)}"}
    
    async def _validate_schema_references(self, sql: str, schema_context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that referenced tables and columns exist"""
        
        try:
            # Extract table references
            table_pattern = r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)'
            tables = re.findall(table_pattern, sql, re.IGNORECASE)
            
            join_pattern = r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)'
            join_tables = re.findall(join_pattern, sql, re.IGNORECASE)
            
            all_tables = tables + join_tables
            
            # Get available tables from schema context
            available_tables = schema_context.get("relevant_tables", [])
            available_table_names = [t.name for t in available_tables] if available_tables else []
            
            # Check table existence
            for table in all_tables:
                table_name = table.split('.')[-1]  # Get table name without schema
                if table_name not in available_table_names:
                    return {
                        "is_valid": False,
                        "error": f"Table '{table_name}' not found or not accessible"
                    }
            
            return {"is_valid": True}
            
        except Exception as e:
            return {"is_valid": False, "error": f"Schema validation error: {str(e)}"}
    
    async def _analyze_query_performance(self, sql: str) -> Dict[str, Any]:
        """Analyze query for performance characteristics"""
        
        warnings = []
        estimated_rows = None
        
        sql_upper = sql.upper()
        
        # Check for performance anti-patterns
        if 'SELECT *' in sql_upper:
            warnings.append("SELECT * may retrieve unnecessary columns")
        
        if 'ORDER BY' in sql_upper and 'LIMIT' not in sql_upper:
            warnings.append("ORDER BY without LIMIT may be expensive")
        
        if sql_upper.count('JOIN') > 3:
            warnings.append("Query has many JOINs which may impact performance")
        
        # Estimate row count (simple heuristic)
        if 'LIMIT' in sql_upper:
            limit_match = re.search(r'LIMIT\s+(\d+)', sql_upper)
            if limit_match:
                estimated_rows = int(limit_match.group(1))
        else:
            # Default estimate based on query complexity
            estimated_rows = 1000  # Conservative default
        
        return {
            "warnings": warnings,
            "estimated_rows": estimated_rows
        }
    
    async def _estimate_query_cost(self, sql: str, schema_context: Dict[str, Any]) -> float:
        """Estimate query execution cost"""
        
        # Simple cost model based on:
        # - Number of tables
        # - Expected row count
        # - JOIN complexity
        
        base_cost = 0.01  # Base cost in dollars
        
        # Table cost
        table_count = len(re.findall(r'\b(FROM|JOIN)\s+', sql, re.IGNORECASE))
        table_cost = table_count * 0.005
        
        # JOIN cost
        join_count = len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))
        join_cost = join_count * 0.01
        
        # Aggregation cost
        agg_count = len(re.findall(r'\b(COUNT|SUM|AVG|MIN|MAX)\b', sql, re.IGNORECASE))
        agg_cost = agg_count * 0.002
        
        total_cost = base_cost + table_cost + join_cost + agg_cost
        
        return round(total_cost, 4)
    
    async def _get_execution_plan(self, sql: str) -> Dict[str, Any]:
        """Get query execution plan (dry run)"""
        
        try:
            # Use EXPLAIN to get execution plan
            explain_sql = f"EXPLAIN {sql}"
            
            # This would execute against the actual database
            # For now, return a mock plan
            return {
                "plan_type": "explain",
                "estimated_cost": 1.0,
                "estimated_rows": 1000,
                "operations": ["TableScan", "Filter", "Sort"]
            }
            
        except Exception as e:
            return {"error": f"Could not generate execution plan: {str(e)}"}
    
    async def _should_sample_query(self, sql: str) -> bool:
        """Determine if query should be sampled for performance"""
        
        sql_upper = sql.upper()
        
        # Sample if no LIMIT clause and potential for large results
        if 'LIMIT' not in sql_upper:
            return True
        
        # Sample if complex aggregations without WHERE clause
        if ('GROUP BY' in sql_upper or 'ORDER BY' in sql_upper) and 'WHERE' not in sql_upper:
            return True
        
        return False
    
    async def _add_sampling_clause(self, sql: str, sample_size: int) -> str:
        """Add sampling to query for large result sets"""
        
        # For Snowflake: use SAMPLE clause
        # For other databases, use different sampling methods
        
        if 'SAMPLE' not in sql.upper() and 'LIMIT' not in sql.upper():
            # Add TABLESAMPLE for Snowflake
            # This is a simplified approach - would need database-specific logic
            return f"{sql} LIMIT {sample_size}"
        
        return sql
    
    async def _add_row_limit(self, sql: str, max_rows: int) -> str:
        """Add row limit to query for safety"""
        
        if 'LIMIT' not in sql.upper():
            return f"{sql} LIMIT {max_rows}"
        
        return sql
    
    async def _execute_with_timeout(self, sql: str, timeout: int, job_id: str) -> Dict[str, Any]:
        """Execute SQL with timeout protection"""
        
        try:
            # This would execute against the actual database with timeout
            # For now, return mock data
            
            # Simulate execution
            await asyncio.sleep(0.1)  # Simulate DB call
            
            # Mock successful result
            mock_data = [
                {"id": 1, "name": "Sample Row 1", "value": 100},
                {"id": 2, "name": "Sample Row 2", "value": 200},
                {"id": 3, "name": "Sample Row 3", "value": 300}
            ]
            
            return {
                "success": True,
                "data": mock_data,
                "columns": ["id", "name", "value"],
                "cost": 0.05
            }
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Query timed out after {timeout} seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _apply_data_masking(self, data: List[Dict[str, Any]], user_id: str) -> List[Dict[str, Any]]:
        """Apply data masking based on user permissions"""
        
        # Check user's PII access permissions
        can_access_pii = await self.rbac_manager.can_access_pii(user_id)
        
        if can_access_pii:
            return data  # No masking needed
        
        # Apply masking to PII fields
        masked_data = []
        pii_patterns = ["email", "phone", "ssn", "credit_card"]
        
        for row in data:
            masked_row = {}
            for key, value in row.items():
                if any(pattern in key.lower() for pattern in pii_patterns):
                    # Mask PII data
                    if isinstance(value, str) and len(value) > 4:
                        masked_row[key] = value[:2] + "*" * (len(value) - 4) + value[-2:]
                    else:
                        masked_row[key] = "***"
                else:
                    masked_row[key] = value
            masked_data.append(masked_row)
        
        return masked_data
    
    def _generate_job_id(self, sql: str, user_id: str) -> str:
        """Generate unique job ID for query tracking"""
        
        timestamp = str(int(time.time() * 1000))
        sql_hash = hashlib.md5(sql.encode()).hexdigest()[:8]
        
        return f"job_{timestamp}_{sql_hash}"
