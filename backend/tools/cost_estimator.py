"""
Cost Estimator - Query cost and resource usage estimation
Provides cost, resource, and performance estimates for governance and approval workflows
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CostEstimate:
    estimated_cost: float
    estimated_rows: int
    estimated_runtime: float
    resource_usage: Dict[str, Any]
    cost_breakdown: Dict[str, float]
    warnings: List[str]
    approval_required: bool = False
    notes: List[str] = None

class CostEstimator:
    """
    Enterprise-grade cost estimation for SQL queries
    """
    def __init__(self):
        # Default cost factors (can be customized per DB)
        self.base_cost = 0.01  # Base cost in dollars
        self.row_cost = 0.00001  # Per row
        self.join_cost = 0.005  # Per join
        self.agg_cost = 0.002  # Per aggregation
        self.complexity_cost = 0.01  # For complex queries
        self.max_cost_threshold = 100.0  # Approval required above this
    
    async def estimate(
        self, 
        sql: str, 
        schema_context: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> CostEstimate:
        """
        Estimate cost and resource usage for a query
        """
        warnings = []
        notes = []
        cost_breakdown = {}
        
        # Table and join analysis
        table_count = len(re.findall(r'\b(FROM|JOIN)\s+', sql, re.IGNORECASE))
        join_count = len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))
        agg_count = len(re.findall(r'\b(COUNT|SUM|AVG|MIN|MAX)\b', sql, re.IGNORECASE))
        
        # Estimate rows (simple heuristic)
        estimated_rows = 1000
        if 'LIMIT' in sql.upper():
            limit_match = re.search(r'LIMIT\s+(\d+)', sql.upper())
            if limit_match:
                estimated_rows = int(limit_match.group(1))
        
        # Estimate runtime (simple heuristic)
        estimated_runtime = 0.5 + 0.01 * estimated_rows + 0.2 * join_count
        
        # Cost calculation
        cost = self.base_cost
        cost += table_count * 0.005
        cost += join_count * self.join_cost
        cost += agg_count * self.agg_cost
        cost += estimated_rows * self.row_cost
        if join_count > 2 or agg_count > 2:
            cost += self.complexity_cost
            warnings.append("High query complexity detected")
        
        cost_breakdown = {
            "base": self.base_cost,
            "tables": table_count * 0.005,
            "joins": join_count * self.join_cost,
            "aggregations": agg_count * self.agg_cost,
            "rows": estimated_rows * self.row_cost,
            "complexity": self.complexity_cost if join_count > 2 or agg_count > 2 else 0.0
        }
        
        approval_required = cost > self.max_cost_threshold
        if approval_required:
            warnings.append(f"Estimated cost ${cost:.2f} exceeds approval threshold")
            notes.append("Approval required for high-cost query")
        
        resource_usage = {
            "tables": table_count,
            "joins": join_count,
            "aggregations": agg_count,
            "estimated_rows": estimated_rows,
            "estimated_runtime": estimated_runtime
        }
        
        return CostEstimate(
            estimated_cost=round(cost, 4),
            estimated_rows=estimated_rows,
            estimated_runtime=round(estimated_runtime, 2),
            resource_usage=resource_usage,
            cost_breakdown=cost_breakdown,
            warnings=warnings,
            approval_required=approval_required,
            notes=notes
        )
