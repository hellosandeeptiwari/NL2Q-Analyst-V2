"""
Agentic Query Orchestrator - Core planning and execution engine
Implements the agent loop: Plan → Tool Selection → Validation → Execution → Refinement
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from datetime import datetime
import hashlib
import os

from backend.tools.schema_tool import SchemaTool
from backend.tools.semantic_dictionary import SemanticDictionary
from backend.tools.sql_runner import SQLRunner
from backend.tools.chart_builder import ChartBuilder
from backend.tools.pii_mask import PIIMask
from backend.tools.cost_estimator import CostEstimator
from backend.governance.rbac_manager import RBACManager
from backend.tools.query_cache import QueryCache
from backend.audit.audit_logger import AuditLogger

class PlanStatus(Enum):
    DRAFT = "draft"
    VALIDATED = "validated" 
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_APPROVAL = "requires_approval"

class ToolType(Enum):
    SCHEMA_DISCOVERY = "schema_discovery"
    SEMANTIC_MAPPING = "semantic_mapping"
    SQL_GENERATION = "sql_generation"
    PYTHON_GENERATION = "python_generation"
    VALIDATION = "validation"
    EXECUTION = "execution"
    VISUALIZATION = "visualization"
    PII_REDACTION = "pii_redaction"
    COST_ESTIMATION = "cost_estimation"

@dataclass
class ExecutionStep:
    step_id: str
    tool_type: ToolType
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    cost: float = 0.0

@dataclass
class QueryPlan:
    plan_id: str
    user_query: str
    reasoning_steps: List[str]
    execution_steps: List[ExecutionStep]
    status: PlanStatus
    created_at: datetime
    user_id: str
    session_id: str
    context: Dict[str, Any]
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    cache_key: Optional[str] = None
    plan_type: str = "agentic_query"
    
    def add_step(self, tool_type: ToolType, input_data: Dict[str, Any]) -> str:
        step_id = str(uuid.uuid4())
        step = ExecutionStep(
            step_id=step_id,
            tool_type=tool_type,
            input_data=input_data
        )
        self.execution_steps.append(step)
        return step_id
    
    def get_step(self, step_id: str) -> Optional[ExecutionStep]:
        return next((step for step in self.execution_steps if step.step_id == step_id), None)
    
    def to_dict(self) -> Dict[str, Any]:
        def serialize_step(step: ExecutionStep) -> Dict[str, Any]:
            step_dict = asdict(step)
            step_dict['tool_type'] = step.tool_type.value
            if step.start_time:
                step_dict['start_time'] = step.start_time.isoformat()
            if step.end_time:
                step_dict['end_time'] = step.end_time.isoformat()
            return step_dict
        
        return {
            'plan_id': self.plan_id,
            'user_query': self.user_query,
            'reasoning_steps': self.reasoning_steps,
            'execution_steps': [serialize_step(step) for step in self.execution_steps],
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'context': self.context,
            'estimated_cost': self.estimated_cost,
            'actual_cost': self.actual_cost,
            'cache_key': self.cache_key,
            'plan_type': self.plan_type
        }

class AgenticOrchestrator:
    """
    Core agentic orchestrator that manages the complete query lifecycle:
    1. Natural language understanding
    2. Plan generation 
    3. Tool orchestration
    4. Validation and governance
    5. Execution and result rendering
    """
    
    def __init__(self):
        # Initialize tools
        self.schema_tool = SchemaTool()
        self.semantic_dict = SemanticDictionary()
        self.sql_runner = SQLRunner()
        self.chart_builder = ChartBuilder()
        self.pii_mask = PIIMask()
        self.cost_estimator = CostEstimator()
        
        # Initialize governance and caching
        self.rbac_manager = RBACManager()
        self.query_cache = QueryCache()
        self.audit_logger = AuditLogger()
        
        # Model configuration
        self.fast_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.reasoning_model = os.getenv("REASONING_MODEL", "o3-mini")
        self.use_reasoning = os.getenv("USE_REASONING_FOR_PLANNING", "true").lower() == "true"
        self.reasoning_temperature = float(os.getenv("REASONING_MODEL_TEMPERATURE", "0.1"))
        
        # Active plans tracking
        self.active_plans: Dict[str, QueryPlan] = {}
        
    async def process_query(
        self, 
        user_query: str, 
        user_id: str, 
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """
        Main entry point for processing a natural language query
        """
        
        # Generate cache key for identical queries
        cache_key = self._generate_cache_key(user_query, user_id, context or {})
        
        # Check cache first
        cached_result = self.query_cache.get(user_query, user_id)
        if cached_result:
            await self.audit_logger.log_cache_hit(user_id, user_query, cache_key)
            return self._create_cached_plan(cached_result, user_query, user_id, session_id)
        
        # Create new plan
        plan = QueryPlan(
            plan_id=str(uuid.uuid4()),
            user_query=user_query,
            reasoning_steps=[],
            execution_steps=[],
            status=PlanStatus.DRAFT,
            created_at=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            context=context or {},
            cache_key=cache_key
        )
        
        self.active_plans[plan.plan_id] = plan
        
        try:
            # Phase 1: Planning and reasoning
            await self._generate_reasoning_plan(plan)
            
            # Phase 2: Schema discovery and semantic mapping
            await self._discover_schema_context(plan)
            
            # Phase 3: Cost estimation and governance check
            await self._estimate_and_validate_cost(plan)
            
            # Phase 4: SQL/Code generation
            await self._generate_query_code(plan)
            
            # Phase 5: Python code generation 
            await self._generate_python_code(plan)
            
            # Phase 6: Validation and safety checks
            await self._validate_query(plan)
            
            # Phase 7: Execution (with approval if needed)
            if plan.status != PlanStatus.REQUIRES_APPROVAL:
                await self._execute_query(plan)
                
            # Phase 8: Visualization and result formatting
            if plan.status == PlanStatus.COMPLETED:
                await self._generate_visualizations(plan)
                
            # Cache successful results
            if plan.status == PlanStatus.COMPLETED:
                self.query_cache.set(
                    query=plan.user_query,
                    result=plan.to_dict(),
                    user_id=plan.user_id,
                    lineage={"plan_id": plan.plan_id, "execution_steps": len(plan.execution_steps)}
                )
                
        except Exception as e:
            plan.status = PlanStatus.FAILED
            await self.audit_logger.log_error(user_id, plan.plan_id, str(e))
            
        finally:
            # Log audit trail
            await self.audit_logger.log_plan_execution(plan)
            
        return plan
    
    async def _generate_reasoning_plan(self, plan: QueryPlan):
        """Generate step-by-step reasoning for the query"""
        
        reasoning_prompt = f"""
        Analyze this natural language query and create a step-by-step reasoning plan:
        Query: "{plan.user_query}"
        
        Consider:
        1. What data entities are mentioned?
        2. What filters or aggregations are needed?
        3. What time periods or date ranges?
        4. What output format would be most useful?
        5. Any potential data governance concerns?
        """
        
        # Add reasoning step to plan
        step_id = plan.add_step(ToolType.SEMANTIC_MAPPING, {
            "query": plan.user_query,
            "prompt": reasoning_prompt
        })
        
        step = plan.get_step(step_id)
        step.start_time = datetime.now()
        
        try:
            # Use reasoning model for complex semantic analysis if enabled
            if self.use_reasoning:
                print(f"🧠 Using {self.reasoning_model} for deep reasoning analysis")
                reasoning_result = await self.semantic_dict.analyze_query_with_reasoning(
                    plan.user_query, 
                    model=self.reasoning_model, 
                    temperature=self.reasoning_temperature
                )
            else:
                print(f"⚡ Using {self.fast_model} for standard analysis")
                reasoning_result = await self.semantic_dict.analyze_query(plan.user_query)
            
            plan.reasoning_steps = reasoning_result.reasoning_steps
            step.output_data = {
                "intent": reasoning_result.intent,
                "entities": reasoning_result.entities,
                "filters": reasoning_result.filters,
                "aggregations": reasoning_result.aggregations,
                "time_dimension": reasoning_result.time_dimension,
                "output_format": reasoning_result.output_format,
                "complexity_score": reasoning_result.complexity_score,
                "reasoning_steps": reasoning_result.reasoning_steps
            }
            step.status = "completed"
            
            # Store entities in plan context for later steps
            plan.context["entities"] = reasoning_result.entities
            plan.context["query_intent"] = reasoning_result.intent
            
        except Exception as e:
            step.error = str(e)
            step.status = "failed"
            raise
        finally:
            step.end_time = datetime.now()
    
    async def _discover_schema_context(self, plan: QueryPlan):
        """Discover relevant schema elements for the query"""
        
        step_id = plan.add_step(ToolType.SCHEMA_DISCOVERY, {
            "entities": plan.context.get("entities", []),
            "reasoning_steps": plan.reasoning_steps
        })
        
        step = plan.get_step(step_id)
        step.start_time = datetime.now()
        
        try:
            # Use schema tool to find relevant tables and columns
            schema_context = await self.schema_tool.discover_schema(
                query=plan.user_query,
                entities=plan.context.get("entities", [])
            )
            
            step.output_data = schema_context.to_dict()
            step.status = "completed"
            
            # Store schema context in plan for later steps
            plan.context["schema_context"] = schema_context.to_dict()
            
        except Exception as e:
            step.error = str(e)
            step.status = "failed"
            raise
        finally:
            step.end_time = datetime.now()
    
    async def _estimate_and_validate_cost(self, plan: QueryPlan):
        """Estimate query cost and check governance policies"""
        
        step_id = plan.add_step(ToolType.COST_ESTIMATION, {
            "schema_context": plan.context.get("schema_context", {}),
            "user_id": plan.user_id
        })
        
        step = plan.get_step(step_id)
        step.start_time = datetime.now()
        
        try:
            # Estimate cost
            cost_estimate = await self.cost_estimator.estimate(
                sql="",  # We don't have SQL yet, estimate based on query
                schema_context=plan.context.get("schema_context", {}),
                user_context={"query": plan.user_query, "user_id": plan.user_id}
            )
            
            plan.estimated_cost = cost_estimate.estimated_cost
            
            # Check governance policies
            governance_check = await self.rbac_manager.check_query_permissions(
                user_id=plan.user_id,
                sql="",  # SQL will be generated in the next phase
                estimated_cost=plan.estimated_cost
            )
            
            if governance_check.get("requires_approval", False):
                plan.status = PlanStatus.REQUIRES_APPROVAL
                
            step.output_data = {
                "cost_estimate": {
                    "estimated_cost": cost_estimate.estimated_cost,
                    "estimated_rows": cost_estimate.estimated_rows,
                    "estimated_runtime": cost_estimate.estimated_runtime,
                    "resource_usage": cost_estimate.resource_usage,
                    "cost_breakdown": cost_estimate.cost_breakdown,
                    "warnings": cost_estimate.warnings,
                    "approval_required": cost_estimate.approval_required,
                    "notes": cost_estimate.notes or []
                },
                "governance_check": governance_check
            }
            step.status = "completed"
            
        except Exception as e:
            step.error = str(e)
            step.status = "failed"
            raise
        finally:
            step.end_time = datetime.now()
    
    async def _generate_query_code(self, plan: QueryPlan):
        """Generate SQL or Python code for the query"""
        
        step_id = plan.add_step(ToolType.SQL_GENERATION, {
            "user_query": plan.user_query,
            "schema_context": plan.context.get("schema_context", {}),
            "reasoning_steps": plan.reasoning_steps
        })
        
        step = plan.get_step(step_id)
        step.start_time = datetime.now()
        
        try:
            # Generate SQL using enhanced generator
            from backend.nl2sql.enhanced_generator import generate_enhanced_sql
            
            sql_result = await generate_enhanced_sql(
                natural_language=plan.user_query,
                schema_context=plan.context.get("schema_context", {}),
                user_context={"user_id": plan.user_id}
            )
            
            step.output_data = sql_result
            step.status = "completed"
            
            # Store generated SQL in plan context
            plan.context["generated_sql"] = sql_result.get("sql", "")
            plan.context["sql_explanation"] = sql_result.get("explanation", "")
            
        except Exception as e:
            step.error = str(e)
            step.status = "failed"
            raise
        finally:
            step.end_time = datetime.now()
    
    async def _generate_python_code(self, plan: QueryPlan):
        """Generate Python/pandas code for data analysis"""
        
        step_id = plan.add_step(ToolType.PYTHON_GENERATION, {
            "user_query": plan.user_query,
            "generated_sql": plan.context.get("generated_sql", ""),
            "schema_context": plan.context.get("schema_context", {}),
            "reasoning_steps": plan.reasoning_steps
        })
        
        step = plan.get_step(step_id)
        step.start_time = datetime.now()
        
        try:
            # Generate Python code based on the SQL and query intent
            python_prompt = f"""
            Generate Python/pandas code to work with the SQL query results and perform additional analysis.
            
            Original User Query: {plan.user_query}
            Generated SQL: {plan.context.get("generated_sql", "")}
            
            Create Python code that:
            1. Loads the SQL results into a pandas DataFrame
            2. Performs any additional data transformations needed
            3. Calculates relevant metrics or aggregations
            4. Prepares the data for visualization
            5. Includes basic data validation and error handling
            
            Return clean Python code that follows best practices.
            """
            
            # Use the same model as SQL generation for consistency
            import openai
            response = await openai.chat.completions.acreate(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are an expert Python data analyst. Generate clean, efficient Python/pandas code for data analysis."},
                    {"role": "user", "content": python_prompt}
                ],
                max_completion_tokens=800,
                temperature=0.1
            )
            
            python_code = response.choices[0].message.content.strip()
            
            # Clean up Python code formatting
            if python_code.startswith("```python"):
                python_code = python_code[9:]
            elif python_code.startswith("```"):
                python_code = python_code[3:]
            if python_code.endswith("```"):
                python_code = python_code[:-3]
            python_code = python_code.strip()
            
            step.output_data = {
                "python_code": python_code,
                "sql_query": plan.context.get("generated_sql", ""),
                "analysis_type": "data_processing"
            }
            step.status = "completed"
            
            # Store generated Python code in plan context
            plan.context["generated_python"] = python_code
            plan.context["python_explanation"] = "Python code for data analysis and processing"
            
        except Exception as e:
            step.error = str(e)
            step.status = "failed"
            raise
        finally:
            step.end_time = datetime.now()
    
    async def _validate_query(self, plan: QueryPlan):
        """Perform validation checks on generated query"""
        
        step_id = plan.add_step(ToolType.VALIDATION, {
            "sql": plan.context.get("generated_sql", ""),
            "schema_context": plan.context.get("schema_context", {})
        })
        
        step = plan.get_step(step_id)
        step.start_time = datetime.now()
        
        try:
            validation_result = await self.sql_runner.validate_query(
                sql=plan.context.get("generated_sql", ""),
                schema_context=plan.context.get("schema_context", {})
            )
            
            if not validation_result.get("is_valid", False):
                step.error = validation_result.get("error", "Validation failed")
                step.status = "failed"
                plan.status = PlanStatus.FAILED
                return
                
            step.output_data = validation_result
            step.status = "completed"
            plan.status = PlanStatus.VALIDATED
            
        except Exception as e:
            step.error = str(e)
            step.status = "failed"
            raise
        finally:
            step.end_time = datetime.now()
    
    async def _execute_query(self, plan: QueryPlan):
        """Execute the validated query"""
        
        step_id = plan.add_step(ToolType.EXECUTION, {
            "sql": plan.context.get("generated_sql", ""),
            "user_id": plan.user_id
        })
        
        step = plan.get_step(step_id)
        step.start_time = datetime.now()
        plan.status = PlanStatus.EXECUTING
        
        try:
            # Execute with timeout and limits
            execution_result = await self.sql_runner.execute_query(
                sql=plan.context.get("generated_sql", ""),
                user_id=plan.user_id,
                timeout_seconds=30,
                max_rows=10000
            )
            
            # Apply PII masking if needed
            if execution_result.get("contains_pii", False):
                masked_result = await self.pii_mask.apply_masking(
                    data=execution_result.get("data", []),
                    user_id=plan.user_id
                )
                execution_result["data"] = masked_result
            
            step.output_data = execution_result
            step.cost = execution_result.get("cost", 0)
            plan.actual_cost += step.cost
            step.status = "completed"
            plan.status = PlanStatus.COMPLETED
            
            # Store results in plan context
            plan.context["query_results"] = execution_result
            
        except Exception as e:
            step.error = str(e)
            step.status = "failed"
            plan.status = PlanStatus.FAILED
            raise
        finally:
            step.end_time = datetime.now()
    
    async def _generate_visualizations(self, plan: QueryPlan):
        """Generate appropriate visualizations for the results"""
        
        query_results = plan.context.get("query_results", {})
        if not query_results.get("data"):
            return
            
        step_id = plan.add_step(ToolType.VISUALIZATION, {
            "data": query_results.get("data", []),
            "query": plan.user_query
        })
        
        step = plan.get_step(step_id)
        step.start_time = datetime.now()
        
        try:
            # Generate chart recommendations using the ChartBuilder's analyze_and_recommend method
            chart_recommendation = await self.chart_builder.analyze_and_recommend(
                data=query_results.get("data", []),
                query_context={"query": plan.user_query, "entities": plan.context.get("entities", [])}
            )
            
            # Create detailed chart specification
            chart_spec = await self.chart_builder.create_chart_spec(
                data=query_results.get("data", []),
                chart_type=chart_recommendation.chart_type,
                query_context={"query": plan.user_query, "entities": plan.context.get("entities", [])}
            )
            
            viz_result = {
                "recommendation": {
                    "chart_type": chart_recommendation.chart_type,
                    "confidence": chart_recommendation.confidence,
                    "reasoning": chart_recommendation.reasoning,
                    "alternative_charts": chart_recommendation.alternative_charts
                },
                "chart_spec": {
                    "chart_type": chart_spec.chart_type,
                    "title": chart_spec.title,
                    "x_axis": chart_spec.x_axis,
                    "y_axis": chart_spec.y_axis,
                    "data_config": chart_spec.data_config,
                    "layout_config": chart_spec.layout_config,
                    "interactive_features": chart_spec.interactive_features,
                    "accessibility_config": chart_spec.accessibility_config
                },
                "charts": [{
                    "type": chart_spec.chart_type,
                    "title": chart_spec.title,
                    "data": chart_spec.data_config,
                    "config": chart_spec.layout_config
                }]
            }
            
            step.output_data = viz_result
            step.status = "completed"
            
            # Store visualizations in plan context
            plan.context["visualizations"] = viz_result
            
        except Exception as e:
            step.error = str(e)
            step.status = "failed"
            # Don't fail the entire plan for visualization errors
        finally:
            step.end_time = datetime.now()
    
    def _generate_cache_key(self, query: str, user_id: str, context: Dict[str, Any]) -> str:
        """Generate a deterministic cache key for query memoization"""
        cache_data = {
            "query": query.lower().strip(),
            "user_id": user_id,
            "context": json.dumps(context, sort_keys=True)
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _create_cached_plan(self, cached_data: Dict[str, Any], user_query: str, user_id: str, session_id: str) -> QueryPlan:
        """Create a plan object from cached data"""
        plan = QueryPlan(
            plan_id=str(uuid.uuid4()),
            user_query=user_query,
            reasoning_steps=cached_data.get("reasoning_steps", []),
            execution_steps=[],  # Don't re-execute steps for cached results
            status=PlanStatus.COMPLETED,
            created_at=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            context=cached_data.get("context", {}),
            cache_key=cached_data.get("cache_key")
        )
        
        # Mark as cached result
        plan.context["is_cached"] = True
        plan.context["cache_timestamp"] = cached_data.get("created_at")
        
        return plan
    
    async def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a plan"""
        plan = self.active_plans.get(plan_id)
        if not plan:
            return None
            
        return {
            "plan_id": plan.plan_id,
            "status": plan.status.value,
            "progress": len([s for s in plan.execution_steps if s.status == "completed"]) / len(plan.execution_steps) if plan.execution_steps else 0,
            "current_step": next((s.tool_type.value for s in plan.execution_steps if s.status == "pending"), None),
            "estimated_cost": plan.estimated_cost,
            "actual_cost": plan.actual_cost
        }
    
    async def approve_plan(self, plan_id: str, approver_id: str) -> bool:
        """Approve a plan that requires human approval"""
        plan = self.active_plans.get(plan_id)
        if not plan or plan.status != PlanStatus.REQUIRES_APPROVAL:
            return False
            
        # Log approval
        self.audit_logger.log_approval(plan_id, approver_id)
        
        # Continue execution
        await self._execute_query(plan)
        if plan.status == PlanStatus.COMPLETED:
            await self._generate_visualizations(plan)
            
        return True
