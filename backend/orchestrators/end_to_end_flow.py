"""
Complete End-to-End User Flow Implementation
Handles: Input → Plan → Generate → Validate → Execute → Render → Iterate
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import re

from backend.agents.enhanced_orchestrator import EnhancedAgenticOrchestrator
from backend.tools.schema_tool import SchemaTool
from backend.tools.semantic_dictionary import SemanticDictionary
from backend.tools.sql_runner import SQLRunner
from backend.tools.chart_builder import ChartBuilder
from backend.tools.query_validator import QueryValidator
from backend.tools.inline_renderer import InlineRenderer
from backend.history.enhanced_chat_history import get_chat_history_manager, MessageType

@dataclass
class UserInput:
    """Structured user input with optional filters"""
    query: str
    filters: Dict[str, Any] = None
    date_range: Dict[str, str] = None
    products: List[str] = None
    cohorts: List[str] = None
    output_format: str = "interactive"  # interactive, table, chart, summary
    max_rows: int = 1000
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = {}

@dataclass
class ExecutionPlan:
    """Detailed execution plan with validation steps"""
    plan_id: str
    reasoning_steps: List[str]
    tool_sequence: List[Dict[str, Any]]
    validation_checks: List[str]
    estimated_cost: float
    estimated_runtime: int  # seconds
    safety_level: str  # low, medium, high
    requires_approval: bool = False

@dataclass
class QueryResult:
    """Rich query result with inline rendering"""
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    visualizations: List[Dict[str, Any]]
    narrative_summary: str
    provenance: Dict[str, Any]
    refinement_suggestions: List[Dict[str, Any]]
    download_links: Dict[str, str]
    execution_stats: Dict[str, Any]

class EndToEndFlowOrchestrator:
    """
    Complete end-to-end flow orchestrator implementing all requirements:
    1. Input parsing with filters
    2. Intelligent planning
    3. Safe code generation
    4. Multi-layer validation
    5. Sandboxed execution
    6. Rich inline rendering
    7. Context-aware iteration
    """
    
    def __init__(self):
        self.base_orchestrator = EnhancedAgenticOrchestrator()
        self.schema_tool = SchemaTool()
        self.semantic_dict = SemanticDictionary()
        self.sql_runner = SQLRunner()
        self.chart_builder = ChartBuilder()
        
        # Get connection params from config
        connection_params = {
            "account": "your_account",
            "user": "your_user", 
            "password": "your_password",
            "warehouse": "your_warehouse",
            "database": "your_database",
            "schema": "your_schema"
        }
        self.query_validator = QueryValidator(connection_params)
        self.inline_renderer = InlineRenderer()
        self.chat_manager = get_chat_history_manager()
        
        # Business synonym mappings for pharma
        self.pharma_synonyms = {
            "writers": ["prescribers", "hcp_count", "physician_count"],
            "nbrx": ["new_prescriptions", "new_rx", "new_scripts"],
            "lapsed": ["inactive_patients", "discontinued_patients"],
            "msl": ["medical_science_liaison", "field_medical"],
            "hcp": ["healthcare_provider", "physician", "prescriber"],
            "nps": ["net_promoter_score", "satisfaction_score"],
            "adherence": ["persistence", "compliance", "medication_adherence"],
            "market_share": ["market_position", "share_of_voice"],
            "therapeutic_area": ["indication", "disease_area", "therapy_area"]
        }
    
    async def process_user_input(
        self, 
        raw_input: str, 
        user_id: str, 
        session_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """
        Main entry point for end-to-end processing
        """
        
        # Step 1: Parse and structure input
        user_input = await self._parse_user_input(raw_input, filters)
        
        # Step 2: Generate execution plan
        plan = await self._create_execution_plan(user_input, user_id, session_id)
        
        # Step 3: Validate plan and queries
        validation_result = await self._validate_plan(plan, user_id)
        
        if not validation_result["is_valid"]:
            raise ValueError(f"Plan validation failed: {validation_result['errors']}")
        
        # Step 4: Execute in sandbox
        execution_result = await self._execute_plan_safely(plan, user_id, session_id)
        
        # Step 5: Render inline results
        final_result = await self._render_inline_results(
            execution_result, plan, user_input, user_id, session_id
        )
        
        # Step 6: Update conversation context
        await self._update_conversation_context(final_result, user_id, session_id)
        
        return final_result
    
    async def _parse_user_input(
        self, 
        raw_input: str, 
        filters: Optional[Dict[str, Any]] = None
    ) -> UserInput:
        """
        Parse plain English input and extract structured components
        """
        
        # Extract date ranges
        date_range = self._extract_date_range(raw_input)
        
        # Extract product mentions
        products = self._extract_products(raw_input)
        
        # Extract cohort specifications
        cohorts = self._extract_cohorts(raw_input)
        
        # Determine output format preference
        output_format = self._determine_output_format(raw_input)
        
        # Extract row limits
        max_rows = self._extract_row_limit(raw_input)
        
        return UserInput(
            query=raw_input,
            filters=filters or {},
            date_range=date_range,
            products=products,
            cohorts=cohorts,
            output_format=output_format,
            max_rows=max_rows
        )
    
    async def _create_execution_plan(
        self, 
        user_input: UserInput, 
        user_id: str, 
        session_id: str
    ) -> ExecutionPlan:
        """
        Create detailed execution plan with reasoning
        """
        
        reasoning_prompt = f"""
        Create a detailed execution plan for this pharmaceutical analytics query:
        
        USER QUERY: {user_input.query}
        FILTERS: {json.dumps(user_input.filters, indent=2)}
        DATE RANGE: {user_input.date_range}
        PRODUCTS: {user_input.products}
        COHORTS: {user_input.cohorts}
        
        Plan should include:
        1. Schema discovery requirements
        2. Business logic translation 
        3. SQL generation with guardrails
        4. Validation checkpoints
        5. Visualization recommendations
        6. Safety and compliance checks
        
        Focus on pharmaceutical business context and regulatory compliance.
        """
        
        # Use reasoning model for complex planning
        reasoning_steps = await self.base_orchestrator._call_reasoning_model(reasoning_prompt)
        
        # Generate tool sequence
        tool_sequence = [
            {
                "tool": "schema_discovery",
                "params": {
                    "query_entities": self._extract_entities(user_input.query),
                    "required_tables": ["rx_facts", "hcp_master", "product_master"],
                    "date_filter_required": bool(user_input.date_range)
                }
            },
            {
                "tool": "semantic_mapping",
                "params": {
                    "business_terms": self._extract_business_terms(user_input.query),
                    "synonym_mapping": True,
                    "pharma_context": True
                }
            },
            {
                "tool": "sql_generation",
                "params": {
                    "template_based": True,
                    "add_guardrails": True,
                    "limit_rows": user_input.max_rows,
                    "compliance_mode": True
                }
            },
            {
                "tool": "validation",
                "params": {
                    "static_checks": True,
                    "schema_validation": True,
                    "dry_run": True,
                    "cost_estimation": True
                }
            },
            {
                "tool": "execution",
                "params": {
                    "sandbox_mode": True,
                    "timeout": 300,
                    "monitor_resources": True
                }
            },
            {
                "tool": "visualization",
                "params": {
                    "auto_suggest": True,
                    "pharma_templates": True,
                    "interactive": user_input.output_format == "interactive"
                }
            }
        ]
        
        # Estimate cost and runtime
        estimated_cost = await self._estimate_execution_cost(tool_sequence)
        estimated_runtime = await self._estimate_runtime(tool_sequence)
        
        # Determine safety level
        safety_level = self._assess_safety_level(user_input, tool_sequence)
        
        return ExecutionPlan(
            plan_id=f"plan_{int(time.time())}",
            reasoning_steps=reasoning_steps,
            tool_sequence=tool_sequence,
            validation_checks=[
                "SQL syntax validation",
                "Schema reference validation", 
                "Permission checking",
                "Resource usage estimation",
                "PII/PHI detection",
                "Business logic verification"
            ],
            estimated_cost=estimated_cost,
            estimated_runtime=estimated_runtime,
            safety_level=safety_level,
            requires_approval=safety_level == "high"
        )
    
    async def _validate_plan(self, plan: ExecutionPlan, user_id: str) -> Dict[str, Any]:
        """
        Multi-layer validation of execution plan
        """
        
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "checks_passed": []
        }
        
        # 1. Static checks
        static_result = await self.query_validator.validate_static(plan)
        validation_results["checks_passed"].append("static_validation")
        
        # 2. Schema checks  
        schema_result = await self.query_validator.validate_schema(plan)
        validation_results["checks_passed"].append("schema_validation")
        
        # 3. Dry run / EXPLAIN
        try:
            dry_run_result = await self.query_validator.dry_run(plan)
            validation_results["checks_passed"].append("dry_run")
        except Exception as e:
            validation_results["errors"].append(f"Dry run failed: {str(e)}")
            validation_results["is_valid"] = False
        
        # 4. Cost and row limit checks
        if plan.estimated_cost > 10.0:  # $10 threshold
            validation_results["warnings"].append(f"High estimated cost: ${plan.estimated_cost:.2f}")
        
        # 5. Permission validation
        permission_result = await self._validate_user_permissions(plan, user_id)
        if not permission_result["allowed"]:
            validation_results["errors"].append("Insufficient permissions")
            validation_results["is_valid"] = False
        
        return validation_results
    
    async def _execute_plan_safely(
        self, 
        plan: ExecutionPlan, 
        user_id: str, 
        session_id: str
    ) -> Dict[str, Any]:
        """
        Execute plan in secure sandbox with monitoring
        """
        
        execution_context = {
            "plan_id": plan.plan_id,
            "user_id": user_id,
            "session_id": session_id,
            "start_time": datetime.now(),
            "timeout": plan.estimated_runtime * 2,  # 2x buffer
            "resource_limits": {
                "max_memory_mb": 1024,
                "max_cpu_percent": 50,
                "max_execution_time": 300
            }
        }
        
        try:
            # Execute each tool in sequence
            results = {}
            
            for step in plan.tool_sequence:
                tool_name = step["tool"]
                params = step["params"]
                
                if tool_name == "schema_discovery":
                    results["schema"] = await self.schema_tool.discover_relevant_schema(
                        params["query_entities"], 
                        params.get("required_tables", [])
                    )
                
                elif tool_name == "semantic_mapping":
                    results["semantic"] = await self.semantic_dict.map_business_terms(
                        params["business_terms"],
                        results.get("schema", {}),
                        pharma_context=params.get("pharma_context", False)
                    )
                
                elif tool_name == "sql_generation":
                    results["sql"] = await self._generate_safe_sql(
                        results["schema"],
                        results["semantic"], 
                        params
                    )
                
                elif tool_name == "execution":
                    results["data"] = await self.sql_runner.execute_with_monitoring(
                        results["sql"],
                        context=execution_context
                    )
                
                elif tool_name == "visualization":
                    results["visualizations"] = await self.chart_builder.build_pharma_visualizations(
                        results["data"],
                        params.get("pharma_templates", True)
                    )
            
            execution_context["end_time"] = datetime.now()
            execution_context["success"] = True
            
            return {
                "results": results,
                "execution_context": execution_context,
                "performance_stats": self._calculate_performance_stats(execution_context)
            }
            
        except Exception as e:
            execution_context["end_time"] = datetime.now()
            execution_context["success"] = False
            execution_context["error"] = str(e)
            
            # Log execution failure
            await self._log_execution_failure(execution_context, e)
            
            raise
    
    async def _render_inline_results(
        self,
        execution_result: Dict[str, Any],
        plan: ExecutionPlan,
        user_input: UserInput,
        user_id: str,
        session_id: str
    ) -> QueryResult:
        """
        Render rich inline results with all components
        """
        
        results = execution_result["results"]
        
        # 1. Format data for display
        formatted_data = await self.inline_renderer.format_table_data(
            results.get("data", []),
            max_preview_rows=200,
            include_pagination=True
        )
        
        # 2. Generate visualizations
        visualizations = results.get("visualizations", [])
        
        # 3. Create narrative summary
        narrative = await self._generate_narrative_summary(
            results.get("data", []),
            user_input.query,
            execution_result["performance_stats"]
        )
        
        # 4. Build provenance information
        provenance = {
            "data_sources": results.get("schema", {}).get("tables_used", []),
            "sql_query": results.get("sql", ""),
            "execution_cost": execution_result["performance_stats"].get("cost", 0),
            "last_updated": datetime.now().isoformat(),
            "execution_time_ms": execution_result["performance_stats"].get("duration_ms", 0),
            "rows_returned": len(results.get("data", [])),
            "plan_id": plan.plan_id
        }
        
        # 5. Generate refinement suggestions
        refinement_suggestions = await self._generate_refinement_suggestions(
            user_input, results, execution_result
        )
        
        # 6. Create download links
        download_links = await self._create_download_links(
            results.get("data", []),
            plan.plan_id
        )
        
        return QueryResult(
            data=formatted_data,
            metadata={
                "total_rows": len(results.get("data", [])),
                "columns": list(results.get("data", [{}])[0].keys()) if results.get("data") else [],
                "execution_plan": plan.plan_id,
                "user_input": asdict(user_input)
            },
            visualizations=visualizations,
            narrative_summary=narrative,
            provenance=provenance,
            refinement_suggestions=refinement_suggestions,
            download_links=download_links,
            execution_stats=execution_result["performance_stats"]
        )
    
    # Helper methods for parsing and extraction
    def _extract_date_range(self, query: str) -> Optional[Dict[str, str]]:
        """Extract date ranges from natural language"""
        date_patterns = {
            r"last (\d+) weeks?": lambda m: {
                "start": (datetime.now() - timedelta(weeks=int(m.group(1)))).strftime("%Y-%m-%d"),
                "end": datetime.now().strftime("%Y-%m-%d")
            },
            r"q(\d) (\d{4})": lambda m: {
                "start": f"{m.group(2)}-{(int(m.group(1))-1)*3+1:02d}-01",
                "end": f"{m.group(2)}-{int(m.group(1))*3:02d}-30"
            },
            r"last quarter": lambda m: {
                "start": (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                "end": datetime.now().strftime("%Y-%m-%d")
            }
        }
        
        query_lower = query.lower()
        for pattern, extractor in date_patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                return extractor(match)
        
        return None
    
    def _extract_products(self, query: str) -> List[str]:
        """Extract product names from query"""
        # This would use NER or domain-specific matching
        common_products = ["humira", "keytruda", "ozempic", "januvia", "lipitor"]
        products = []
        
        query_lower = query.lower()
        for product in common_products:
            if product in query_lower:
                products.append(product)
        
        return products
    
    def _extract_cohorts(self, query: str) -> List[str]:
        """Extract patient/HCP cohorts from query"""
        cohort_patterns = {
            r"age (\d+)-(\d+)": "age_group",
            r"new patients": "new_patients",
            r"specialty": "specialty_hcp",
            r"primary care": "primary_care_hcp"
        }
        
        cohorts = []
        query_lower = query.lower()
        for pattern, cohort_type in cohort_patterns.items():
            if re.search(pattern, query_lower):
                cohorts.append(cohort_type)
        
        return cohorts
    
    def _determine_output_format(self, query: str) -> str:
        """Determine preferred output format"""
        if any(word in query.lower() for word in ["chart", "graph", "plot", "visual"]):
            return "chart"
        elif any(word in query.lower() for word in ["table", "list", "data"]):
            return "table"
        elif any(word in query.lower() for word in ["summary", "overview", "insights"]):
            return "summary"
        else:
            return "interactive"
    
    def _extract_row_limit(self, query: str) -> int:
        """Extract row limit from query"""
        limit_match = re.search(r"top (\d+)|limit (\d+)|first (\d+)", query.lower())
        if limit_match:
            return int(limit_match.group(1) or limit_match.group(2) or limit_match.group(3))
        return 1000  # Default
    
    async def _generate_narrative_summary(
        self,
        data: List[Dict[str, Any]],
        original_query: str,
        performance_stats: Dict[str, Any]
    ) -> str:
        """Generate key takeaways narrative"""
        
        if not data:
            return "No data returned for the specified criteria."
        
        summary_parts = []
        
        # Basic stats
        summary_parts.append(f"Retrieved {len(data)} records")
        
        # Top/bottom analysis
        if len(data) > 1:
            # Find numeric columns for analysis
            numeric_cols = []
            for col, value in data[0].items():
                if isinstance(value, (int, float)):
                    numeric_cols.append(col)
            
            if numeric_cols:
                # Analyze first numeric column
                main_metric = numeric_cols[0]
                values = [row[main_metric] for row in data if row[main_metric] is not None]
                
                if values:
                    max_val = max(values)
                    min_val = min(values)
                    
                    # Find records with max/min values
                    max_record = next(row for row in data if row[main_metric] == max_val)
                    min_record = next(row for row in data if row[main_metric] == min_val)
                    
                    # Create identifier for records (first non-numeric column)
                    id_col = next((col for col, val in max_record.items() if not isinstance(val, (int, float))), None)
                    
                    if id_col:
                        summary_parts.append(
                            f"Highest {main_metric}: {max_record[id_col]} ({max_val:,.0f})"
                        )
                        summary_parts.append(
                            f"Lowest {main_metric}: {min_record[id_col]} ({min_val:,.0f})"
                        )
        
        # Performance note
        if performance_stats.get("duration_ms", 0) > 5000:
            summary_parts.append(f"Query completed in {performance_stats['duration_ms']/1000:.1f} seconds")
        
        return " • ".join(summary_parts) + "."

    def _extract_entities(self, query: str) -> List[str]:
        """Extract key entities from the query for schema discovery"""
        entities = []
        
        # Common NBA/pharma entities
        nba_patterns = [
            r'\bnba\b', r'\bbasket\b', r'\boutput\b', r'\brecommend\w*\b',
            r'\bprovider\b', r'\bphysician\b', r'\bhcp\b', r'\bmessage\b',
            r'\btherapeutic\b', r'\barea\b', r'\bscore\b', r'\bfinal\b'
        ]
        
        query_lower = query.lower()
        for pattern in nba_patterns:
            import re
            matches = re.findall(pattern, query_lower)
            entities.extend(matches)
        
        # Remove duplicates while preserving order
        unique_entities = []
        for entity in entities:
            if entity not in unique_entities:
                unique_entities.append(entity)
        
        return unique_entities

    def _extract_business_terms(self, query: str) -> List[str]:
        """Extract business terms that need semantic mapping"""
        business_terms = []
        
        # Pharma business terms
        pharma_terms = [
            'prescriber', 'physician', 'hcp', 'provider', 'doctor',
            'prescription', 'rx', 'nbrx', 'script',
            'patient', 'therapy', 'treatment', 'therapeutic',
            'market share', 'adherence', 'compliance',
            'recommendation', 'message', 'engagement',
            'analytics', 'insights', 'score', 'ranking'
        ]
        
        query_lower = query.lower()
        for term in pharma_terms:
            if term in query_lower:
                business_terms.append(term)
        
        return business_terms

    async def _estimate_execution_cost(self, tool_sequence: List[Dict[str, Any]]) -> float:
        """Estimate cost of executing the tool sequence"""
        base_cost = 0.001  # Base cost in dollars
        
        for tool in tool_sequence:
            tool_name = tool.get("tool", "")
            if tool_name == "schema_discovery":
                base_cost += 0.002  # Schema discovery cost
            elif tool_name == "semantic_mapping":
                base_cost += 0.003  # LLM embedding cost
            elif tool_name == "sql_generation":
                base_cost += 0.005  # SQL generation cost
            elif tool_name == "execution":
                base_cost += 0.010  # Database execution cost
            elif tool_name == "visualization":
                base_cost += 0.002  # Chart generation cost
                
        return base_cost

    async def _estimate_runtime(self, tool_sequence: List[Dict[str, Any]]) -> int:
        """Estimate runtime in seconds"""
        base_time = 5  # Base time in seconds
        
        for tool in tool_sequence:
            tool_name = tool.get("tool", "")
            if tool_name == "schema_discovery":
                base_time += 10  # Schema discovery time
            elif tool_name == "semantic_mapping":
                base_time += 15  # Embedding time
            elif tool_name == "sql_generation":
                base_time += 5   # SQL generation time
            elif tool_name == "execution":
                base_time += 30  # Database execution time
            elif tool_name == "validation":
                base_time += 5   # Validation time
            elif tool_name == "visualization":
                base_time += 10  # Chart generation time
                
        return base_time

    def _assess_safety_level(self, user_input: UserInput, tool_sequence: List[Dict[str, Any]]) -> str:
        """Assess safety level of the execution plan"""
        query_lower = user_input.query.lower()
        
        # High risk indicators
        high_risk_terms = ['delete', 'drop', 'truncate', 'update', 'insert', 'alter']
        if any(term in query_lower for term in high_risk_terms):
            return "high"
            
        # Medium risk indicators  
        medium_risk_terms = ['export', 'download', 'file', 'email', 'send']
        if any(term in query_lower for term in medium_risk_terms):
            return "medium"
            
        # Check for large data requests
        if user_input.max_rows > 10000:
            return "medium"
            
        return "low"
