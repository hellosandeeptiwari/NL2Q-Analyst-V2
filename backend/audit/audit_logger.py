"""
Audit Logger - Observability, compliance, and lineage tracking
Logs query executions, errors, approvals, and lineage for enterprise auditability
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

AUDIT_LOG_PATH = os.path.join(os.path.dirname(__file__), '../audit/audit_log.jsonl')

@dataclass
class AuditEvent:
    event_type: str
    user_id: str
    timestamp: datetime
    details: Dict[str, Any]
    job_id: Optional[str] = None
    status: Optional[str] = None
    lineage: Optional[Dict[str, Any]] = None

class AuditLogger:
    """
    Enterprise-grade audit logger for query executions, errors, approvals, and lineage
    """
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path or AUDIT_LOG_PATH
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
    
    async def log_query_execution(
        self, 
        user_id: str, 
        sql: str, 
        job_id: str,
        row_count: int,
        execution_time: float,
        cost: float,
        lineage: Optional[Dict[str, Any]] = None
    ):
        event = AuditEvent(
            event_type="query_execution",
            user_id=user_id,
            timestamp=datetime.now(),
            details={
                "sql": sql,
                "row_count": row_count,
                "execution_time": execution_time,
                "cost": cost
            },
            job_id=job_id,
            status="success",
            lineage=lineage
        )
        await self._write_event(event)
    
    async def log_query_error(
        self, 
        user_id: str, 
        sql: str, 
        job_id: str,
        error: str,
        execution_time: float
    ):
        event = AuditEvent(
            event_type="query_error",
            user_id=user_id,
            timestamp=datetime.now(),
            details={
                "sql": sql,
                "error": error,
                "execution_time": execution_time
            },
            job_id=job_id,
            status="error"
        )
        await self._write_event(event)
    
    async def log_approval(
        self, 
        user_id: str, 
        request_id: str,
        approved: bool,
        approver_id: str,
        reason: str
    ):
        event = AuditEvent(
            event_type="approval",
            user_id=user_id,
            timestamp=datetime.now(),
            details={
                "request_id": request_id,
                "approved": approved,
                "approver_id": approver_id,
                "reason": reason
            },
            status="approved" if approved else "denied"
        )
        await self._write_event(event)
    
    async def log_lineage(
        self, 
        user_id: str, 
        job_id: str,
        lineage: Dict[str, Any]
    ):
        event = AuditEvent(
            event_type="lineage",
            user_id=user_id,
            timestamp=datetime.now(),
            details={},
            job_id=job_id,
            lineage=lineage
        )
        await self._write_event(event)
    
    async def log_error(
        self, 
        user_id: str, 
        plan_id: str, 
        error_message: str
    ):
        """Log an error event"""
        event = AuditEvent(
            event_type="error",
            user_id=user_id,
            timestamp=datetime.now(),
            details={
                "plan_id": plan_id,
                "error_message": error_message
            },
            job_id=plan_id,
            status="error"
        )
        await self._write_event(event)
    
    async def log_plan_execution(self, plan):
        """Log the complete plan execution"""
        event = AuditEvent(
            event_type="plan_execution",
            user_id=plan.user_id,
            timestamp=datetime.now(),
            details={
                "plan_id": plan.plan_id,
                "status": plan.status.value,
                "estimated_cost": plan.estimated_cost,
                "actual_cost": plan.actual_cost,
                "execution_steps": len(plan.execution_steps)
            },
            job_id=plan.plan_id,
            status=plan.status.value
        )
        await self._write_event(event)
    
    async def log_cache_hit(self, user_id: str, query: str, cache_key: str):
        """Log a cache hit event"""
        event = AuditEvent(
            event_type="cache_hit",
            user_id=user_id,
            timestamp=datetime.now(),
            details={
                "query": query,
                "cache_key": cache_key
            },
            status="success"
        )
        await self._write_event(event)
    
    async def _write_event(self, event: AuditEvent):
        # Serialize event to JSONL
        event_dict = {
            "event_type": event.event_type,
            "user_id": event.user_id,
            "timestamp": event.timestamp.isoformat(),
            "details": event.details,
            "job_id": event.job_id,
            "status": event.status,
            "lineage": event.lineage
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_dict) + "\n")
