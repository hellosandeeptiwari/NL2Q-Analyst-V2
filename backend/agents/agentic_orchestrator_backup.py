"""
Enhanced Agentic Query Orchestrator - 11-Step NL2SQL Pipeline
Implements the complete agent workflow as per instructions:
0) Contract validation
1) Connect & Read Schema  
2) Build Schema Vector Index
3) Parse User Task
4) Retrieve Relevant Schema (Similarity Search)
5) Plan the Steps
6) Generate SQL (First Draft)
7) Lint Before Execution
8) Execute & Preview
9) Repair Loop (If Execution Fails)
10) Export
11) Visualization (Only if Asked)
"""

import asyncio
import json
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from datetime import datetime, timedelta
import hashlib

from backend.tools.schema_tool import SchemaTool
from backend.tools.semantic_dictionary import SemanticDictionary
from backend.tools.sql_runner import SQLRunner
from backend.tools.chart_builder import ChartBuilder
from backend.tools.pii_mask import PIIMask
from backend.tools.cost_estimator import CostEstimator
from backend.governance.rbac_manager import RBACManager
from backend.tools.query_cache import QueryCache
from backend.audit.audit_logger import AuditLogger

class PipelineStage(Enum):
    """11-step pipeline stages"""
    CONTRACT_VALIDATION = "contract_validation"
    SCHEMA_CONNECTION = "schema_connection"
    VECTOR_INDEXING = "vector_indexing"
    TASK_PARSING = "task_parsing"
    SCHEMA_RETRIEVAL = "schema_retrieval"
    STEP_PLANNING = "step_planning"
    SQL_GENERATION = "sql_generation"
    SQL_LINTING = "sql_linting"
    EXECUTION_PREVIEW = "execution_preview"
    REPAIR_LOOP = "repair_loop"
    EXPORT_GENERATION = "export_generation"
    VISUALIZATION = "visualization"

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
    VALIDATION = "validation"
    EXECUTION = "execution"
    VISUALIZATION = "visualization"
    PII_REDACTION = "pii_redaction"
    COST_ESTIMATION = "cost_estimation"

@dataclass
class TaskComponents:
    """Parsed user task components"""
    metrics: List[str]
    filters: List[Dict[str, Any]]
    time_range: Optional[Dict[str, Any]]
    granularity: Optional[str]
    group_by: List[str]
    sort_by: List[Dict[str, Any]]
    limit: Optional[int]
    visualization_request: Optional[str]
    synonyms: Dict[str, str]

@dataclass
class SchemaCard:
    """Schema object representation"""
    schema_name: str
    table_name: str
    column_name: Optional[str]
    data_type: Optional[str]
    comment: Optional[str]
    sample_values: List[str]
    card_type: str  # 'table' or 'column'
    embedding: Optional[List[float]] = None

@dataclass
class SchemaWhitelist:
    """Validated schema objects for SQL generation"""
    tables: List[str]  # SCHEMA.TABLE format
    columns: List[str]  # SCHEMA.TABLE.COLUMN format
    joins: List[Dict[str, Any]]  # FK relationships

@dataclass
class ExecutionStep:
    step_id: str
    stage: PipelineStage
    tool_type: ToolType
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    cost: float = 0.0
    retry_count: int = 0
    max_retries: int = 3

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
    
    # New fields for 11-step pipeline
    task_components: Optional[TaskComponents] = None
    schema_whitelist: Optional[SchemaWhitelist] = None
    sql_drafts: List[str] = None
    repair_attempts: int = 0
    max_repair_attempts: int = 3
    export_links: List[str] = None
    visualization_code: Optional[str] = None
    
    def __post_init__(self):
        if self.sql_drafts is None:
            self.sql_drafts = []
        if self.export_links is None:
            self.export_links = []
    
    def add_step(self, stage: PipelineStage, tool_type: ToolType, input_data: Dict[str, Any]) -> str:
        step_id = str(uuid.uuid4())
        step = ExecutionStep(
            step_id=step_id,
            stage=stage,
            tool_type=tool_type,
            input_data=input_data
        )
        self.execution_steps.append(step)
        return step_id
    
    def get_step(self, step_id: str) -> Optional[ExecutionStep]:
        return next((step for step in self.execution_steps if step.step_id == step_id), None)
    
    def get_current_stage(self) -> PipelineStage:
        """Get the current pipeline stage"""
        if not self.execution_steps:
            return PipelineStage.CONTRACT_VALIDATION
        
        # Find last completed step
        for step in reversed(self.execution_steps):
            if step.status == "completed":
                # Return next stage in sequence
                stages = list(PipelineStage)
                current_idx = stages.index(step.stage)
                if current_idx < len(stages) - 1:
                    return stages[current_idx + 1]
                return step.stage
        
        return PipelineStage.CONTRACT_VALIDATION
    
    def add_sql_draft(self, sql: str, attempt_number: int):
        """Add a SQL draft with attempt tracking"""
        self.sql_drafts.append({
            'sql': sql,
            'attempt': attempt_number,
            'timestamp': datetime.now().isoformat()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['status'] = self.status.value
        result['created_at'] = self.created_at.isoformat()
        
        # Convert execution steps
        result['execution_steps'] = []
        for step in self.execution_steps:
            step_dict = asdict(step)
            step_dict['stage'] = step.stage.value
            step_dict['tool_type'] = step.tool_type.value
            if step.start_time:
                step_dict['start_time'] = step.start_time.isoformat()
            if step.end_time:
                step_dict['end_time'] = step.end_time.isoformat()
            result['execution_steps'].append(step_dict)
        
        return result

class AgenticOrchestrator:
    """
    Enhanced Agentic Orchestrator implementing the complete 11-step NL2SQL pipeline:
    
    0) Contract validation - Ensure read-only SELECT queries only
    1) Connect & Read Schema - Introspect database structure
    2) Build Schema Vector Index - Embed table/column cards
    3) Parse User Task - Extract metrics, filters, etc.
    4) Retrieve Relevant Schema - Similarity search for relevant objects
    5) Plan the Steps - Create execution roadmap
    6) Generate SQL - First draft with whitelist constraints
    7) Lint Before Execution - Validate safety and syntax
    8) Execute & Preview - Run with timeout and limits
    9) Repair Loop - Fix errors with max 3 attempts
    10) Export - Generate download links if needed
    11) Visualization - Create charts if requested
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
        
        # Active plans tracking
        self.active_plans: Dict[str, QueryPlan] = {}
        
        # Configuration
        self.DEFAULT_LIMIT = 1000
        self.MAX_ROWS_PREVIEW = 100
        self.QUERY_TIMEOUT = 60
        self.DEFAULT_TIME_WINDOW = 90  # days
        
        # SQL safety patterns
        self.FORBIDDEN_PATTERNS = [
            r'\bINSERT\b', r'\bUPDATE\b', r'\bDELETE\b', r'\bMERGE\b',
            r'\bDROP\b', r'\bALTER\b', r'\bTRUNCATE\b', r'\bCALL\b',
            r'\bGRANT\b', r'\bREVOKE\b', r'\bCREATE\b'
        ]
    
    async def process_query(
        self, 
        user_query: str, 
        user_id: str, 
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """
        Main entry point - Execute the complete 11-step pipeline
        """
        
        # Generate cache key for identical queries
        cache_key = self._generate_cache_key(user_query, user_id, context or {})
        
        # Check cache first
        cached_result = await self.query_cache.get(cache_key)
        if cached_result:
            self.audit_logger.log_cache_hit(user_id, user_query, cache_key)
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
    
    # ==================== PIPELINE STEP IMPLEMENTATIONS ====================
    
    async def _step_0_contract_validation(self, plan: QueryPlan):
        """Step 0: Contract validation - Ensure read-only SELECT queries only"""
        step_id = plan.add_step(
            PipelineStage.CONTRACT_VALIDATION,
            ToolType.VALIDATION,
            {"user_query": plan.user_query}
        )
        
        step = plan.get_step(step_id)
        step.start_time = datetime.now()
        step.status = "running"
        
        plan.reasoning_steps.append("üîç Step 0: Validating query contract (read-only)")
        
        try:
            # Check for forbidden SQL operations
            query_upper = plan.user_query.upper()
            for pattern in self.FORBIDDEN_PATTERNS:
                if re.search(pattern, query_upper):
                    raise ValueError(f"Query contains forbidden operation: {pattern}")
            
            # Check user permissions
            permissions = await self.rbac_manager.get_user_permissions(plan.user_id)
            if not permissions.get('can_read_data', False):
                raise ValueError("User does not have data read permissions")
            
            step.output_data = {
                "validation_passed": True,
                "user_permissions": permissions
            }
            step.status = "completed"
            plan.reasoning_steps.append("‚úÖ Contract validated: Read-only access confirmed")
            
        except Exception as e:
            step.error = str(e)
            step.status = "failed"
            plan.status = PlanStatus.FAILED
            plan.reasoning_steps.append(f"‚ùå Contract validation failed: {str(e)}")
            raise
        finally:
            step.end_time = datetime.now()
    
    async def _step_1_schema_connection(self, plan: QueryPlan):
        """Step 1: Connect & Read Schema - Introspect database structure"""
        step_id = plan.add_step(
            PipelineStage.SCHEMA_CONNECTION,
            ToolType.SCHEMA_DISCOVERY,
            {"action": "introspect"}
        )
        
        step = plan.get_step(step_id)
        step.start_time = datetime.now()
        step.status = "running"
        
        plan.reasoning_steps.append("üîó Step 1: Connecting to database and reading schema")
        
        try:
            # Get schema information
            schema_info = await self.schema_tool.get_enhanced_schema()
            
            # Create schema cards
            schema_cards = []
            
            for table_info in schema_info.get('tables', []):
                schema_name = table_info.get('schema', 'public')
                table_name = table_info['name']
                
                # Table card
                table_card = SchemaCard(
                    schema_name=schema_name,
                    table_name=table_name,
                    column_name=None,
                    data_type=None,
                    comment=table_info.get('comment', ''),
                    sample_values=[],
                    card_type='table'
                )
                schema_cards.append(table_card)
                
                # Column cards
                for column in table_info.get('columns', []):
                    column_card = SchemaCard(
                        schema_name=schema_name,
                        table_name=table_name,
                        column_name=column['name'],
                        data_type=column.get('type', ''),
                        comment=column.get('comment', ''),
                        sample_values=column.get('sample_values', [])[:10],  # Max 10 samples
                        card_type='column'
                    )
                    schema_cards.append(column_card)
            
            step.output_data = {
                "schema_info": schema_info,
                "schema_cards": [asdict(card) for card in schema_cards],
                "total_tables": len(schema_info.get('tables', [])),
                "total_columns": sum(len(t.get('columns', [])) for t in schema_info.get('tables', []))
            }
            step.status = "completed"
            
            plan.reasoning_steps.append(
                f"‚úÖ Schema connected: {step.output_data['total_tables']} tables, "
                f"{step.output_data['total_columns']} columns discovered"
            )
            
        except Exception as e:
            step.error = str(e)
            step.status = "failed"
            plan.status = PlanStatus.FAILED
            plan.reasoning_steps.append(f"‚ùå Schema connection failed: {str(e)}")
            raise
        finally:
            step.end_time = datetime.now()
    
    async def _step_2_vector_indexing(self, plan: QueryPlan):
        """Step 2: Build Schema Vector Index - Embed table/column cards"""
        step_id = plan.add_step(
            PipelineStage.VECTOR_INDEXING,
            ToolType.SEMANTIC_MAPPING,
            {"action": "build_index"}
        )
        
        step = plan.get_step(step_id)
        step.start_time = datetime.now()
        step.status = "running"
        
        plan.reasoning_steps.append("üß† Step 2: Building vector index for schema objects")
        
        try:
            # Get schema cards from previous step
            schema_step = next(s for s in plan.execution_steps if s.stage == PipelineStage.SCHEMA_CONNECTION)
            schema_cards = schema_step.output_data['schema_cards']
            
            # Build embeddings for each schema card
            embeddings_built = 0
            for card_data in schema_cards:
                # Create text representation for embedding
                if card_data['card_type'] == 'table':
                    text = f"Table: {card_data['schema_name']}.{card_data['table_name']} - {card_data['comment']}"
                else:
                    text = f"Column: {card_data['schema_name']}.{card_data['table_name']}.{card_data['column_name']} ({card_data['data_type']}) - {card_data['comment']}"
                    if card_data['sample_values']:
                        text += f" Examples: {', '.join(card_data['sample_values'][:3])}"
                
                # Get embedding from semantic dictionary
                embedding = await self.semantic_dict.get_embedding(text)
                card_data['embedding'] = embedding
                embeddings_built += 1
            
            # Store in vector index
            await self.semantic_dict.build_schema_index(schema_cards)
            
            step.output_data = {
                "embeddings_built": embeddings_built,
                "index_size": len(schema_cards)
            }
            step.status = "completed"
            
            plan.reasoning_steps.append(
                f"‚úÖ Vector index built: {embeddings_built} schema objects embedded"
            )
            
        except Exception as e:
            step.error = str(e)
            step.status = "failed"
            plan.status = PlanStatus.FAILED
            plan.reasoning_steps.append(f"‚ùå Vector indexing failed: {str(e)}")
            raise
        finally:
            step.end_time = datetime.now()
    
    async def _step_3_task_parsing(self, plan: QueryPlan):
        """Step 3: Parse User Task - Extract metrics, filters, time range, etc."""
        step_id = plan.add_step(
            PipelineStage.TASK_PARSING,
            ToolType.SEMANTIC_MAPPING,
            {"user_query": plan.user_query}
        )
        
        step = plan.get_step(step_id)
        step.start_time = datetime.now()
        step.status = "running"
        
        plan.reasoning_steps.append("üìù Step 3: Parsing user task components")
        
        try:
            # Parse query using semantic dictionary
            parsed_components = await self.semantic_dict.parse_query_components(plan.user_query)
            
            # Create TaskComponents object
            task_components = TaskComponents(
                metrics=parsed_components.get('metrics', []),
                filters=parsed_components.get('filters', []),
                time_range=parsed_components.get('time_range'),
                granularity=parsed_components.get('granularity'),
                group_by=parsed_components.get('group_by', []),
                sort_by=parsed_components.get('sort_by', []),
                limit=parsed_components.get('limit'),
                visualization_request=parsed_components.get('visualization'),
                synonyms=parsed_components.get('synonyms', {})
            )
            
            # Add default time range if not specified
            if not task_components.time_range:
                task_components.time_range = {
                    'start_date': (datetime.now() - timedelta(days=self.DEFAULT_TIME_WINDOW)).isoformat(),
                    'end_date': datetime.now().isoformat(),
                    'is_default': True
                }
            
            plan.task_components = task_components
            
            step.output_data = {
                "task_components": asdict(task_components),
                "parsing_confidence": parsed_components.get('confidence', 0.8)
            }
            step.status = "completed"
            
            plan.reasoning_steps.append(
                f"‚úÖ Task parsed: {len(task_components.metrics)} metrics, "
                f"{len(task_components.filters)} filters identified"
            )
            
        except Exception as e:
            step.error = str(e)
            step.status = "failed"
            plan.status = PlanStatus.FAILED
            plan.reasoning_steps.append(f"‚ùå Task parsing failed: {str(e)}")
            raise
        finally:
            step.end_time = datetime.now()
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
            
            # Phase 5: Validation and safety checks
            await self._validate_query(plan)
            
            # Phase 6: Execution (with approval if needed)
            if plan.status != PlanStatus.REQUIRES_APPROVAL:
                await self._execute_query(plan)
                
            # Phase 7: Visualization and result formatting
            if plan.status == PlanStatus.COMPLETED:
                await self._generate_visualizations(plan)
                
            # Cache successful results
            if plan.status == PlanStatus.COMPLETED:
                await self.query_cache.set(cache_key, plan.to_dict())
                
        except Exception as e:
            plan.status = PlanStatus.FAILED
            self.audit_logger.log_error(user_id, plan.plan_id, str(e))
            
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
            # Use semantic dictionary to understand the query
            reasoning_result = await self.semantic_dict.analyze_query(plan.user_query)
            
            plan.reasoning_steps = reasoning_result.get("reasoning_steps", [])
            step.output_data = reasoning_result
            step.status = "completed"
            
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
            
            step.output_data = schema_context
            step.status = "completed"
            
            # Store schema context in plan for later steps
            plan.context["schema_context"] = schema_context
            
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
            cost_estimate = await self.cost_estimator.estimate_query_cost(
                schema_context=plan.context.get("schema_context", {}),
                user_query=plan.user_query
            )
            
            plan.estimated_cost = cost_estimate.get("estimated_cost", 0)
            
            # Check governance policies
            governance_check = await self.rbac_manager.check_query_permissions(
                user_id=plan.user_id,
                schema_context=plan.context.get("schema_context", {}),
                estimated_cost=plan.estimated_cost
            )
            
            if governance_check.get("requires_approval", False):
                plan.status = PlanStatus.REQUIRES_APPROVAL
                
            step.output_data = {
                "cost_estimate": cost_estimate,
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
            # Generate charts and insights
            viz_result = await self.chart_builder.auto_generate_charts(
                data=query_results.get("data", []),
                query_context=plan.user_query
            )
            
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
        a s y n c   d e f   _ s t e p _ 4 _ s c h e m a _ r e t r i e v a l ( s e l f ,   p l a n :   Q u e r y P l a n ) :  
                 " " " S t e p   4 :   R e t r i e v e   R e l e v a n t   S c h e m a   -   S i m i l a r i t y   s e a r c h   f o r   r e l e v a n t   o b j e c t s " " "  
                 s t e p _ i d   =   p l a n . a d d _ s t e p (  
                         P i p e l i n e S t a g e . S C H E M A _ R E T R I E V A L ,  
                         T o o l T y p e . S E M A N T I C _ M A P P I N G ,  
                         { " t a s k _ c o m p o n e n t s " :   a s d i c t ( p l a n . t a s k _ c o m p o n e n t s ) }  
                 )  
                  
                 s t e p   =   p l a n . g e t _ s t e p ( s t e p _ i d )  
                 s t e p . s t a r t _ t i m e   =   d a t e t i m e . n o w ( )  
                 s t e p . s t a t u s   =   " r u n n i n g "  
                  
                 p l a n . r e a s o n i n g _ s t e p s . a p p e n d ( "  x ç   S t e p   4 :   R e t r i e v i n g   r e l e v a n t   s c h e m a   o b j e c t s " )  
                  
                 t r y :  
                         #   C r e a t e   s e a r c h   q u e r y   f r o m   t a s k   c o m p o n e n t s  
                         s e a r c h _ q u e r y   =   "   " . j o i n ( [  
                                 p l a n . u s e r _ q u e r y ,  
                                 "   " . j o i n ( p l a n . t a s k _ c o m p o n e n t s . m e t r i c s ) ,  
                                 "   " . j o i n ( p l a n . t a s k _ c o m p o n e n t s . g r o u p _ b y )  
                         ] )  
                          
                         #   G e t   r e l e v a n t   s c h e m a   o b j e c t s   v i a   s i m i l a r i t y   s e a r c h  
                         r e l e v a n t _ o b j e c t s   =   a w a i t   s e l f . s e m a n t i c _ d i c t . s e a r c h _ s c h e m a _ s i m i l a r i t y (  
                                 s e a r c h _ q u e r y ,    
                                 t o p _ t a b l e s = 1 0 ,    
                                 t o p _ c o l u m n s = 5 0  
                         )  
                          
                         #   F i l t e r   t o   o n l y   e x i s t i n g   o b j e c t s  
                         s c h e m a _ s t e p   =   n e x t ( s   f o r   s   i n   p l a n . e x e c u t i o n _ s t e p s   i f   s . s t a g e   = =   P i p e l i n e S t a g e . S C H E M A _ C O N N E C T I O N )  
                         s c h e m a _ i n f o   =   s c h e m a _ s t e p . o u t p u t _ d a t a [ ' s c h e m a _ i n f o ' ]  
                          
                         #   B u i l d   w h i t e l i s t   o f   v a l i d   o b j e c t s  
                         v a l i d _ t a b l e s   =   [ ]  
                         v a l i d _ c o l u m n s   =   [ ]  
                         j o i n s   =   [ ]  
                          
                         #   V a l i d a t e   t a b l e s   e x i s t  
                         f o r   t a b l e _ m a t c h   i n   r e l e v a n t _ o b j e c t s . g e t ( ' t a b l e s ' ,   [ ] ) :  
                                 t a b l e _ f u l l _ n a m e   =   f " { t a b l e _ m a t c h [ ' s c h e m a ' ] } . { t a b l e _ m a t c h [ ' t a b l e ' ] } "  
                                 i f   s e l f . _ t a b l e _ e x i s t s ( s c h e m a _ i n f o ,   t a b l e _ m a t c h [ ' s c h e m a ' ] ,   t a b l e _ m a t c h [ ' t a b l e ' ] ) :  
                                         v a l i d _ t a b l e s . a p p e n d ( t a b l e _ f u l l _ n a m e )  
                          
                         #   V a l i d a t e   c o l u m n s   e x i s t  
                         f o r   c o l u m n _ m a t c h   i n   r e l e v a n t _ o b j e c t s . g e t ( ' c o l u m n s ' ,   [ ] ) :  
                                 c o l u m n _ f u l l _ n a m e   =   f " { c o l u m n _ m a t c h [ ' s c h e m a ' ] } . { c o l u m n _ m a t c h [ ' t a b l e ' ] } . { c o l u m n _ m a t c h [ ' c o l u m n ' ] } "  
                                 i f   s e l f . _ c o l u m n _ e x i s t s ( s c h e m a _ i n f o ,   c o l u m n _ m a t c h [ ' s c h e m a ' ] ,   c o l u m n _ m a t c h [ ' t a b l e ' ] ,   c o l u m n _ m a t c h [ ' c o l u m n ' ] ) :  
                                         v a l i d _ c o l u m n s . a p p e n d ( c o l u m n _ f u l l _ n a m e )  
                          
                         #   E x t r a c t   F K   r e l a t i o n s h i p s   f o r   j o i n s  
                         f o r   t a b l e _ i n f o   i n   s c h e m a _ i n f o . g e t ( ' t a b l e s ' ,   [ ] ) :  
                                 f o r   f k   i n   t a b l e _ i n f o . g e t ( ' f o r e i g n _ k e y s ' ,   [ ] ) :  
                                         j o i n s . a p p e n d ( {  
                                                 ' f r o m _ t a b l e ' :   f " { t a b l e _ i n f o . g e t ( ' s c h e m a ' ,   ' p u b l i c ' ) } . { t a b l e _ i n f o [ ' n a m e ' ] } " ,  
                                                 ' f r o m _ c o l u m n ' :   f k [ ' c o l u m n ' ] ,  
                                                 ' t o _ t a b l e ' :   f " { f k [ ' r e f e r e n c e d _ s c h e m a ' ] } . { f k [ ' r e f e r e n c e d _ t a b l e ' ] } " ,  
                                                 ' t o _ c o l u m n ' :   f k [ ' r e f e r e n c e d _ c o l u m n ' ]  
                                         } )  
                          
                         #   C r e a t e   s c h e m a   w h i t e l i s t  
                         w h i t e l i s t   =   S c h e m a W h i t e l i s t (  
                                 t a b l e s = v a l i d _ t a b l e s ,  
                                 c o l u m n s = v a l i d _ c o l u m n s ,  
                                 j o i n s = j o i n s  
                         )  
                          
                         p l a n . s c h e m a _ w h i t e l i s t   =   w h i t e l i s t  
                          
                         s t e p . o u t p u t _ d a t a   =   {  
                                 " r e l e v a n t _ t a b l e s " :   l e n ( v a l i d _ t a b l e s ) ,  
                                 " r e l e v a n t _ c o l u m n s " :   l e n ( v a l i d _ c o l u m n s ) ,  
                                 " a v a i l a b l e _ j o i n s " :   l e n ( j o i n s ) ,  
                                 " w h i t e l i s t " :   a s d i c t ( w h i t e l i s t )  
                         }  
                         s t e p . s t a t u s   =   " c o m p l e t e d "  
                          
                         p l a n . r e a s o n i n g _ s t e p s . a p p e n d (  
                                 f " ‚ S&   S c h e m a   r e t r i e v e d :   { l e n ( v a l i d _ t a b l e s ) }   t a b l e s ,   { l e n ( v a l i d _ c o l u m n s ) }   c o l u m n s   w h i t e l i s t e d "  
                         )  
                          
                 e x c e p t   E x c e p t i o n   a s   e :  
                         s t e p . e r r o r   =   s t r ( e )  
                         s t e p . s t a t u s   =   " f a i l e d "  
                         p l a n . s t a t u s   =   P l a n S t a t u s . F A I L E D  
                         p l a n . r e a s o n i n g _ s t e p s . a p p e n d ( f " ‚ ù R  S c h e m a   r e t r i e v a l   f a i l e d :   { s t r ( e ) } " )  
                         r a i s e  
                 f i n a l l y :  
                         s t e p . e n d _ t i m e   =   d a t e t i m e . n o w ( )  
          
         a s y n c   d e f   _ s t e p _ 5 _ s t e p _ p l a n n i n g ( s e l f ,   p l a n :   Q u e r y P l a n ) :  
                 " " " S t e p   5 :   P l a n   t h e   S t e p s   -   C r e a t e   e x e c u t i o n   r o a d m a p " " "  
                 s t e p _ i d   =   p l a n . a d d _ s t e p (  
                         P i p e l i n e S t a g e . S T E P _ P L A N N I N G ,  
                         T o o l T y p e . V A L I D A T I O N ,  
                         { " w h i t e l i s t " :   a s d i c t ( p l a n . s c h e m a _ w h i t e l i s t ) }  
                 )  
                  
                 s t e p   =   p l a n . g e t _ s t e p ( s t e p _ i d )  
                 s t e p . s t a r t _ t i m e   =   d a t e t i m e . n o w ( )  
                 s t e p . s t a t u s   =   " r u n n i n g "  
                  
                 p l a n . r e a s o n i n g _ s t e p s . a p p e n d ( "  x 9   S t e p   5 :   C r e a t i n g   e x e c u t i o n   p l a n " )  
                  
                 t r y :  
                         #   C r e a t e   m i n i m a l   e x e c u t i o n   p l a n  
                         e x e c u t i o n _ p l a n   =   {  
                                 ' S 1 ' :   ' R A G   r e t r i e v a l   ‚ S& ' ,  
                                 ' S 2 ' :   ' S Q L   d r a f t   g e n e r a t i o n ' ,  
                                 ' S 3 ' :   ' S Q L   l i n t i n g   a n d   v a l i d a t i o n ' ,  
                                 ' S 4 ' :   ' Q u e r y   e x e c u t i o n   w i t h   p r e v i e w ' ,  
                                 ' S 5 ' :   ' E r r o r   r e p a i r   l o o p   ( i f   n e e d e d ) ' ,  
                                 ' S 6 ' :   ' E x p o r t   l i n k   g e n e r a t i o n ' ,  
                                 ' S 7 ' :   ' V i s u a l i z a t i o n   ( i f   r e q u e s t e d ) '  
                         }  
                          
                         #   E s t i m a t e   c o s t s  
                         e s t i m a t e d _ c o s t   =   a w a i t   s e l f . c o s t _ e s t i m a t o r . e s t i m a t e _ q u e r y _ c o s t (  
                                 p l a n . u s e r _ q u e r y ,  
                                 l e n ( p l a n . s c h e m a _ w h i t e l i s t . t a b l e s ) ,  
                                 l e n ( p l a n . s c h e m a _ w h i t e l i s t . c o l u m n s )  
                         )  
                          
                         p l a n . e s t i m a t e d _ c o s t   =   e s t i m a t e d _ c o s t  
                          
                         #   C h e c k   i f   a p p r o v a l   i s   n e e d e d  
                         r e q u i r e s _ a p p r o v a l   =   (  
                                 e s t i m a t e d _ c o s t   >   1 0 . 0   o r     #   H i g h   c o s t  
                                 l e n ( p l a n . s c h e m a _ w h i t e l i s t . t a b l e s )   >   5   o r     #   C o m p l e x   j o i n s  
                                 ' s e n s i t i v e '   i n   p l a n . u s e r _ q u e r y . l o w e r ( )     #   S e n s i t i v e   d a t a  
                         )  
                          
                         i f   r e q u i r e s _ a p p r o v a l :  
                                 p l a n . s t a t u s   =   P l a n S t a t u s . R E Q U I R E S _ A P P R O V A L  
                          
                         s t e p . o u t p u t _ d a t a   =   {  
                                 " e x e c u t i o n _ p l a n " :   e x e c u t i o n _ p l a n ,  
                                 " e s t i m a t e d _ c o s t " :   e s t i m a t e d _ c o s t ,  
                                 " r e q u i r e s _ a p p r o v a l " :   r e q u i r e s _ a p p r o v a l ,  
                                 " c o m p l e x i t y _ s c o r e " :   l e n ( p l a n . s c h e m a _ w h i t e l i s t . t a b l e s )   +   l e n ( p l a n . s c h e m a _ w h i t e l i s t . c o l u m n s )   /   1 0  
                         }  
                         s t e p . s t a t u s   =   " c o m p l e t e d "  
                          
                         i f   r e q u i r e s _ a p p r o v a l :  
                                 p l a n . r e a s o n i n g _ s t e p s . a p p e n d (  
                                         f " ‚ a† Ô ∏ è   P l a n   c r e a t e d :   R e q u i r e s   a p p r o v a l   ( E s t .   c o s t :   $ { e s t i m a t e d _ c o s t : . 2 f } ) "  
                                 )  
                         e l s e :  
                                 p l a n . r e a s o n i n g _ s t e p s . a p p e n d (  
                                         f " ‚ S&   P l a n   c r e a t e d :   R e a d y   f o r   e x e c u t i o n   ( E s t .   c o s t :   $ { e s t i m a t e d _ c o s t : . 2 f } ) "  
                                 )  
                          
                 e x c e p t   E x c e p t i o n   a s   e :  
                         s t e p . e r r o r   =   s t r ( e )  
                         s t e p . s t a t u s   =   " f a i l e d "  
                         p l a n . s t a t u s   =   P l a n S t a t u s . F A I L E D  
                         p l a n . r e a s o n i n g _ s t e p s . a p p e n d ( f " ‚ ù R  S t e p   p l a n n i n g   f a i l e d :   { s t r ( e ) } " )  
                         r a i s e  
                 f i n a l l y :  
                         s t e p . e n d _ t i m e   =   d a t e t i m e . n o w ( )  
          
         a s y n c   d e f   _ s t e p _ 6 _ s q l _ g e n e r a t i o n ( s e l f ,   p l a n :   Q u e r y P l a n ) :  
                 " " " S t e p   6 :   G e n e r a t e   S Q L   -   F i r s t   d r a f t   w i t h   w h i t e l i s t   c o n s t r a i n t s " " "  
                 s t e p _ i d   =   p l a n . a d d _ s t e p (  
                         P i p e l i n e S t a g e . S Q L _ G E N E R A T I O N ,  
                         T o o l T y p e . S Q L _ G E N E R A T I O N ,  
                         {  
                                 " u s e r _ q u e r y " :   p l a n . u s e r _ q u e r y ,  
                                 " t a s k _ c o m p o n e n t s " :   a s d i c t ( p l a n . t a s k _ c o m p o n e n t s ) ,  
                                 " w h i t e l i s t " :   a s d i c t ( p l a n . s c h e m a _ w h i t e l i s t )  
                         }  
                 )  
                  
                 s t e p   =   p l a n . g e t _ s t e p ( s t e p _ i d )  
                 s t e p . s t a r t _ t i m e   =   d a t e t i m e . n o w ( )  
                 s t e p . s t a t u s   =   " r u n n i n g "  
                  
                 p l a n . r e a s o n i n g _ s t e p s . a p p e n d ( "  x ß   S t e p   6 :   G e n e r a t i n g   S Q L   q u e r y " )  
                  
                 t r y :  
                         #   G e n e r a t e   S Q L   u s i n g   t h e   S Q L   r u n n e r   w i t h   c o n s t r a i n t s  
                         s q l _ r e s u l t   =   a w a i t   s e l f . s q l _ r u n n e r . g e n e r a t e _ s q l (  
                                 q u e r y = p l a n . u s e r _ q u e r y ,  
                                 s c h e m a _ w h i t e l i s t = p l a n . s c h e m a _ w h i t e l i s t . t a b l e s   +   p l a n . s c h e m a _ w h i t e l i s t . c o l u m n s ,  
                                 t a s k _ c o m p o n e n t s = p l a n . t a s k _ c o m p o n e n t s ,  
                                 d i a l e c t = " p o s t g r e s q l " ,     #   C o n f i g u r e   a s   n e e d e d  
                                 d e f a u l t _ l i m i t = s e l f . D E F A U L T _ L I M I T  
                         )  
                          
                         g e n e r a t e d _ s q l   =   s q l _ r e s u l t [ ' s q l ' ]  
                          
                         #   A d d   a u t o m a t i c   L I M I T   i f   n o t   p r e s e n t  
                         i f   n o t   r e . s e a r c h ( r ' \ b L I M I T \ b ' ,   g e n e r a t e d _ s q l . u p p e r ( ) ) :  
                                 g e n e r a t e d _ s q l   + =   f "   L I M I T   { s e l f . D E F A U L T _ L I M I T } "  
                          
                         #   E n s u r e   s c h e m a   q u a l i f i c a t i o n  
                         q u a l i f i e d _ s q l   =   s e l f . _ e n s u r e _ s c h e m a _ q u a l i f i c a t i o n ( g e n e r a t e d _ s q l ,   p l a n . s c h e m a _ w h i t e l i s t )  
                          
                         #   S t o r e   f i r s t   d r a f t  
                         p l a n . a d d _ s q l _ d r a f t ( q u a l i f i e d _ s q l ,   a t t e m p t _ n u m b e r = 1 )  
                          
                         s t e p . o u t p u t _ d a t a   =   {  
                                 " g e n e r a t e d _ s q l " :   q u a l i f i e d _ s q l ,  
                                 " g e n e r a t i o n _ m e t h o d " :   s q l _ r e s u l t . g e t ( ' m e t h o d ' ,   ' l l m ' ) ,  
                                 " c o n f i d e n c e _ s c o r e " :   s q l _ r e s u l t . g e t ( ' c o n f i d e n c e ' ,   0 . 8 ) ,  
                                 " h a s _ l i m i t " :   ' L I M I T '   i n   q u a l i f i e d _ s q l . u p p e r ( ) ,  
                                 " i s _ s c h e m a _ q u a l i f i e d " :   T r u e  
                         }  
                         s t e p . s t a t u s   =   " c o m p l e t e d "  
                          
                         p l a n . r e a s o n i n g _ s t e p s . a p p e n d (  
                                 f " ‚ S&   S Q L   g e n e r a t e d :   { l e n ( q u a l i f i e d _ s q l ) }   c h a r a c t e r s ,   "  
                                 f " c o n f i d e n c e :   { s q l _ r e s u l t . g e t ( ' c o n f i d e n c e ' ,   0 . 8 ) : . 2 f } "  
                         )  
                          
                 e x c e p t   E x c e p t i o n   a s   e :  
                         s t e p . e r r o r   =   s t r ( e )  
                         s t e p . s t a t u s   =   " f a i l e d "  
                         p l a n . s t a t u s   =   P l a n S t a t u s . F A I L E D  
                         p l a n . r e a s o n i n g _ s t e p s . a p p e n d ( f " ‚ ù R  S Q L   g e n e r a t i o n   f a i l e d :   { s t r ( e ) } " )  
                         r a i s e  
                 f i n a l l y :  
                         s t e p . e n d _ t i m e   =   d a t e t i m e . n o w ( )  
          
         a s y n c   d e f   _ s t e p _ 7 _ s q l _ l i n t i n g ( s e l f ,   p l a n :   Q u e r y P l a n ) :  
                 " " " S t e p   7 :   L i n t   B e f o r e   E x e c u t i o n   -   V a l i d a t e   s a f e t y   a n d   s y n t a x " " "  
                 s t e p _ i d   =   p l a n . a d d _ s t e p (  
                         P i p e l i n e S t a g e . S Q L _ L I N T I N G ,  
                         T o o l T y p e . V A L I D A T I O N ,  
                         { " s q l " :   p l a n . s q l _ d r a f t s [ - 1 ] [ ' s q l ' ] }  
                 )  
                  
                 s t e p   =   p l a n . g e t _ s t e p ( s t e p _ i d )  
                 s t e p . s t a r t _ t i m e   =   d a t e t i m e . n o w ( )  
                 s t e p . s t a t u s   =   " r u n n i n g "  
                  
                 p l a n . r e a s o n i n g _ s t e p s . a p p e n d ( "  x: ° Ô ∏ è   S t e p   7 :   L i n t i n g   S Q L   f o r   s a f e t y   a n d   s y n t a x " )  
                  
                 t r y :  
                         c u r r e n t _ s q l   =   p l a n . s q l _ d r a f t s [ - 1 ] [ ' s q l ' ]  
                          
                         #   S a f e t y   c h e c k s  
                         s a f e t y _ i s s u e s   =   [ ]  
                          
                         #   1 .   C h e c k   f o r   f o r b i d d e n   o p e r a t i o n s  
                         f o r   p a t t e r n   i n   s e l f . F O R B I D D E N _ P A T T E R N S :  
                                 i f   r e . s e a r c h ( p a t t e r n ,   c u r r e n t _ s q l . u p p e r ( ) ) :  
                                         s a f e t y _ i s s u e s . a p p e n d ( f " C o n t a i n s   f o r b i d d e n   o p e r a t i o n :   { p a t t e r n } " )  
                          
                         #   2 .   E n s u r e   S E L E C T - o n l y  
                         i f   n o t   c u r r e n t _ s q l . s t r i p ( ) . u p p e r ( ) . s t a r t s w i t h ( ' S E L E C T ' )   a n d   n o t   c u r r e n t _ s q l . s t r i p ( ) . u p p e r ( ) . s t a r t s w i t h ( ' W I T H ' ) :  
                                 s a f e t y _ i s s u e s . a p p e n d ( " Q u e r y   m u s t   s t a r t   w i t h   S E L E C T   o r   W I T H " )  
                          
                         #   3 .   C h e c k   s c h e m a   q u a l i f i c a t i o n  
                         u n q u a l i f i e d _ r e f s   =   s e l f . _ c h e c k _ s c h e m a _ q u a l i f i c a t i o n ( c u r r e n t _ s q l )  
                         i f   u n q u a l i f i e d _ r e f s :  
                                 s a f e t y _ i s s u e s . a p p e n d ( f " U n q u a l i f i e d   r e f e r e n c e s :   { ' ,   ' . j o i n ( u n q u a l i f i e d _ r e f s ) } " )  
                          
                         #   4 .   V a l i d a t e   w h i t e l i s t   c o m p l i a n c e  
                         w h i t e l i s t _ v i o l a t i o n s   =   s e l f . _ c h e c k _ w h i t e l i s t _ c o m p l i a n c e ( c u r r e n t _ s q l ,   p l a n . s c h e m a _ w h i t e l i s t )  
                         i f   w h i t e l i s t _ v i o l a t i o n s :  
                                 s a f e t y _ i s s u e s . a p p e n d ( f " W h i t e l i s t   v i o l a t i o n s :   { ' ,   ' . j o i n ( w h i t e l i s t _ v i o l a t i o n s ) } " )  
                          
                         #   5 .   E n s u r e   L I M I T   i s   p r e s e n t  
                         i f   n o t   r e . s e a r c h ( r ' \ b L I M I T \ b ' ,   c u r r e n t _ s q l . u p p e r ( ) ) :  
                                 s a f e t y _ i s s u e s . a p p e n d ( " M i s s i n g   L I M I T   c l a u s e " )  
                          
                         #   S y n t a x   v a l i d a t i o n   u s i n g   S Q L   r u n n e r  
                         s y n t a x _ v a l i d a t i o n   =   a w a i t   s e l f . s q l _ r u n n e r . v a l i d a t e _ s y n t a x ( c u r r e n t _ s q l )  
                          
                         i f   s a f e t y _ i s s u e s   o r   n o t   s y n t a x _ v a l i d a t i o n [ ' v a l i d ' ] :  
                                 r a i s e   V a l u e E r r o r ( f " L i n t i n g   f a i l e d :   { ' ;   ' . j o i n ( s a f e t y _ i s s u e s   +   [ s y n t a x _ v a l i d a t i o n . g e t ( ' e r r o r ' ,   ' ' ) ] ) } " )  
                          
                         s t e p . o u t p u t _ d a t a   =   {  
                                 " l i n t i n g _ p a s s e d " :   T r u e ,  
                                 " s a f e t y _ c h e c k s " :   l e n ( s e l f . F O R B I D D E N _ P A T T E R N S ) ,  
                                 " s y n t a x _ v a l i d " :   s y n t a x _ v a l i d a t i o n [ ' v a l i d ' ] ,  
                                 " w a r n i n g s " :   s y n t a x _ v a l i d a t i o n . g e t ( ' w a r n i n g s ' ,   [ ] )  
                         }  
                         s t e p . s t a t u s   =   " c o m p l e t e d "  
                          
                         p l a n . r e a s o n i n g _ s t e p s . a p p e n d ( " ‚ S&   S Q L   l i n t i n g   p a s s e d :   S a f e   f o r   e x e c u t i o n " )  
                          
                 e x c e p t   E x c e p t i o n   a s   e :  
                         s t e p . e r r o r   =   s t r ( e )  
                         s t e p . s t a t u s   =   " f a i l e d "  
                         p l a n . s t a t u s   =   P l a n S t a t u s . F A I L E D  
                         p l a n . r e a s o n i n g _ s t e p s . a p p e n d ( f " ‚ ù R  S Q L   l i n t i n g   f a i l e d :   { s t r ( e ) } " )  
                         r a i s e  
                 f i n a l l y :  
                         s t e p . e n d _ t i m e   =   d a t e t i m e . n o w ( )  
 