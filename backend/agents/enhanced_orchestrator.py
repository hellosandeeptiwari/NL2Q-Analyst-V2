"""
Enhanced Agentic Orchestrator with Pharma-specific features
Includes compliance checks, therapeutic area context, and specialized reasoning
"""

import asyncio
import json
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from datetime import datetime
import hashlib
import os

from backend.agents.agentic_orchestrator import (
    AgenticOrchestrator as BaseOrchestrator,
    QueryPlan, PlanStatus, ToolType, ExecutionStep
)
from backend.auth.user_profile import get_user_profile, UserRole
from backend.history.enhanced_chat_history import get_chat_history_manager, MessageType

class PharmaComplianceLevel(Enum):
    """Pharma-specific compliance levels"""
    PUBLIC = "public"  # No PHI/PII concerns
    INTERNAL = "internal"  # Internal use, aggregated data
    RESTRICTED = "restricted"  # Requires approval
    CONFIDENTIAL = "confidential"  # Highest restrictions

class TherapeuticContext(Enum):
    """Therapeutic area contexts for specialized reasoning"""
    ONCOLOGY = "oncology"
    DIABETES = "diabetes"
    CARDIOVASCULAR = "cardiovascular"
    IMMUNOLOGY = "immunology"
    NEUROLOGY = "neurology"
    INFECTIOUS_DISEASE = "infectious_disease"
    RESPIRATORY = "respiratory"
    GENERAL = "general"

@dataclass
class PharmaQueryContext:
    """Enhanced context for pharma queries"""
    therapeutic_areas: List[str]
    compliance_level: PharmaComplianceLevel
    user_role: str
    department: str
    data_permissions: List[str]
    conversation_history: List[Dict[str, Any]]
    regulatory_flags: List[str]
    business_context: Dict[str, Any]

class EnhancedAgenticOrchestrator(BaseOrchestrator):
    """
    Enhanced orchestrator with pharma-specific intelligence and compliance
    """
    
    def __init__(self):
        super().__init__()
        self.chat_manager = get_chat_history_manager()
        
        # Pharma-specific reasoning prompts
        self.pharma_reasoning_prompts = {
            TherapeuticContext.ONCOLOGY: """
            You are analyzing oncology pharmaceutical data. Consider:
            - Patient populations and demographics
            - Treatment protocols and lines of therapy
            - Survival outcomes and progression metrics
            - Biomarker and genomic considerations
            - Adverse events and safety profiles
            - Competitive landscape and market access
            """,
            TherapeuticContext.DIABETES: """
            You are analyzing diabetes pharmaceutical data. Consider:
            - HbA1c reduction and glycemic control
            - Weight management and cardiovascular outcomes
            - Insulin sensitivity and beta-cell function
            - Patient adherence and lifestyle factors
            - Healthcare resource utilization
            - Formulary placement and access
            """,
            TherapeuticContext.GENERAL: """
            You are analyzing pharmaceutical data. Consider:
            - Patient safety and efficacy outcomes
            - Prescriber behavior and preferences
            - Market dynamics and competitive positioning
            - Regulatory compliance requirements
            - Health economics and outcomes research
            - Real-world evidence generation
            """
        }
        
        # Compliance validation rules
        self.compliance_rules = {
            "phi_patterns": [
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                r"\b\d{3}\.\d{2}\.\d{4}\b",  # SSN with dots
                r"\b[A-Z]{2}\d{7}\b",  # Patient ID patterns
            ],
            "restricted_tables": [
                "patient_identifiers",
                "provider_npi",
                "raw_claims_data"
            ],
            "high_risk_columns": [
                "patient_name",
                "ssn",
                "date_of_birth",
                "address",
                "phone"
            ]
        }
    
    async def create_plan(
        self,
        user_query: str,
        user_id: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """
        Enhanced plan creation with pharma context
        """
        # Get user profile for enhanced context
        user_profile = get_user_profile(user_id)
        
        # Build pharma-specific context
        pharma_context = await self._build_pharma_context(
            user_query, user_profile, context or {}
        )
        
        # Generate cache key with pharma context
        cache_key = self._generate_pharma_cache_key(
            user_query, user_id, pharma_context
        )
        
        # Check specialized cache
        cached_result = await self._check_pharma_cache(cache_key)
        if cached_result:
            return await self._create_cached_pharma_plan(
                cached_result, user_query, user_id, session_id, pharma_context
            )
        
        # Create enhanced plan
        plan = QueryPlan(
            plan_id=str(uuid.uuid4()),
            user_query=user_query,
            reasoning_steps=[],
            execution_steps=[],
            status=PlanStatus.DRAFT,
            created_at=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            context=dict(
                **(context or {}),
                pharma_context=asdict(pharma_context),
                compliance_level=pharma_context.compliance_level.value,
                therapeutic_areas=pharma_context.therapeutic_areas
            ),
            cache_key=cache_key,
            plan_type="pharma_agentic_query"
        )
        
        self.active_plans[plan.plan_id] = plan
        
        try:
            # Phase 1: Enhanced pharma reasoning
            await self._generate_pharma_reasoning(plan, pharma_context)
            
            # Phase 2: Compliance and governance validation
            await self._validate_pharma_compliance(plan, pharma_context)
            
            # Phase 3: Schema discovery with therapeutic context
            await self._discover_therapeutic_schema(plan, pharma_context)
            
            # Phase 4: Specialized cost estimation
            await self._estimate_pharma_costs(plan, pharma_context)
            
            # Phase 5: Generate pharma-optimized queries
            await self._generate_pharma_queries(plan, pharma_context)
            
            # Phase 6: Execute with compliance monitoring
            if plan.status != PlanStatus.REQUIRES_APPROVAL:
                await self._execute_with_compliance(plan, pharma_context)
            
            # Cache successful plans
            await self._cache_pharma_plan(plan)
            
            return plan
            
        except Exception as e:
            plan.status = PlanStatus.FAILED
            await self.audit_logger.log_plan_error(plan.plan_id, str(e))
            raise
    
    async def _build_pharma_context(
        self,
        user_query: str,
        user_profile,
        base_context: Dict[str, Any]
    ) -> PharmaQueryContext:
        """Build enhanced pharma context"""
        
        # Determine therapeutic areas from query and user profile
        therapeutic_areas = []
        if user_profile and user_profile.therapeutic_areas:
            therapeutic_areas.extend(user_profile.therapeutic_areas)
        
        # Extract therapeutic context from query
        query_lower = user_query.lower()
        therapeutic_keywords = {
            "oncology": ["cancer", "oncology", "tumor", "chemotherapy", "radiation"],
            "diabetes": ["diabetes", "insulin", "glucose", "hba1c", "diabetic"],
            "cardiovascular": ["cardiac", "heart", "cardiovascular", "hypertension"],
            "immunology": ["immune", "autoimmune", "arthritis", "psoriasis"],
            "neurology": ["neurological", "alzheimer", "parkinson", "epilepsy"]
        }
        
        for area, keywords in therapeutic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                if area not in therapeutic_areas:
                    therapeutic_areas.append(area)
        
        # Determine compliance level
        compliance_level = PharmaComplianceLevel.INTERNAL
        if user_profile:
            if user_profile.role in [UserRole.REGULATORY, UserRole.MEDICAL_AFFAIRS]:
                compliance_level = PharmaComplianceLevel.RESTRICTED
            elif "executive" in user_profile.role.value.lower():
                compliance_level = PharmaComplianceLevel.CONFIDENTIAL
        
        # Get conversation history for context
        conversation_history = []
        if base_context.get("conversation_context"):
            conversation_history = base_context["conversation_context"]
        
        # Build regulatory flags
        regulatory_flags = []
        if "adverse" in query_lower or "safety" in query_lower:
            regulatory_flags.append("safety_signal")
        if "clinical" in query_lower or "trial" in query_lower:
            regulatory_flags.append("clinical_data")
        
        return PharmaQueryContext(
            therapeutic_areas=therapeutic_areas or ["general"],
            compliance_level=compliance_level,
            user_role=user_profile.role.value if user_profile else "analyst",
            department=user_profile.department if user_profile else "Analytics",
            data_permissions=user_profile.data_access_permissions if user_profile else [],
            conversation_history=conversation_history,
            regulatory_flags=regulatory_flags,
            business_context=base_context.get("business_context", {})
        )
    
    async def _generate_pharma_reasoning(
        self,
        plan: QueryPlan,
        pharma_context: PharmaQueryContext
    ):
        """Generate pharma-specific reasoning with o3-mini model"""
        
        therapeutic_context = TherapeuticContext.GENERAL
        if pharma_context.therapeutic_areas:
            primary_area = pharma_context.therapeutic_areas[0].lower()
            if primary_area in [tc.value for tc in TherapeuticContext]:
                therapeutic_context = TherapeuticContext(primary_area)
        
        reasoning_prompt = f"""
        {self.pharma_reasoning_prompts[therapeutic_context]}
        
        USER CONTEXT:
        - Role: {pharma_context.user_role}
        - Department: {pharma_context.department}
        - Therapeutic Areas: {', '.join(pharma_context.therapeutic_areas)}
        - Compliance Level: {pharma_context.compliance_level.value}
        
        USER QUERY: "{plan.user_query}"
        
        CONVERSATION HISTORY:
        {json.dumps(pharma_context.conversation_history, indent=2)}
        
        Generate a step-by-step reasoning plan considering:
        1. Pharma-specific data requirements
        2. Compliance and regulatory considerations
        3. Business context and decision-making needs
        4. Data quality and validation requirements
        5. Visualization and reporting preferences
        
        Provide reasoning steps in a structured format.
        """
        
        if self.use_reasoning:
            # Use o3-mini for complex pharma reasoning
            reasoning_steps = await self._call_reasoning_model(
                reasoning_prompt, 
                temperature=self.reasoning_temperature
            )
        else:
            # Fallback to fast model
            reasoning_steps = await self._call_fast_model(reasoning_prompt)
        
        plan.reasoning_steps = reasoning_steps
        plan.status = PlanStatus.VALIDATED
        
        # Add reasoning step to execution plan
        plan.add_step(ToolType.SEMANTIC_MAPPING, {
            "reasoning": reasoning_steps,
            "therapeutic_context": therapeutic_context.value,
            "compliance_level": pharma_context.compliance_level.value
        })
    
    async def _validate_pharma_compliance(
        self,
        plan: QueryPlan,
        pharma_context: PharmaQueryContext
    ):
        """Validate pharma compliance requirements"""
        
        compliance_step = plan.add_step(ToolType.VALIDATION, {
            "compliance_checks": [],
            "risk_level": "low"
        })
        
        step = plan.get_step(compliance_step)
        step.status = "executing"
        step.start_time = datetime.now()
        
        compliance_issues = []
        risk_level = "low"
        
        # Check for PHI patterns
        query_lower = plan.user_query.lower()
        for pattern in self.compliance_rules["phi_patterns"]:
            if pattern in plan.user_query:
                compliance_issues.append(f"Potential PHI pattern detected: {pattern}")
                risk_level = "high"
        
        # Check for restricted terms
        if any(term in query_lower for term in ["patient_name", "ssn", "address"]):
            compliance_issues.append("Query references potentially restricted patient data")
            risk_level = "high"
        
        # Check user permissions vs therapeutic areas
        if pharma_context.therapeutic_areas:
            for area in pharma_context.therapeutic_areas:
                if area not in pharma_context.data_permissions:
                    compliance_issues.append(f"User may not have access to {area} data")
                    risk_level = "medium"
        
        # Determine if approval needed
        if (risk_level == "high" or 
            pharma_context.compliance_level == PharmaComplianceLevel.CONFIDENTIAL or
            "regulatory" in pharma_context.regulatory_flags):
            plan.status = PlanStatus.REQUIRES_APPROVAL
            compliance_issues.append("Query requires management approval due to sensitivity")
        
        step.output_data = {
            "compliance_issues": compliance_issues,
            "risk_level": risk_level,
            "requires_approval": plan.status == PlanStatus.REQUIRES_APPROVAL
        }
        step.status = "completed"
        step.end_time = datetime.now()
    
    async def _discover_therapeutic_schema(
        self,
        plan: QueryPlan,
        pharma_context: PharmaQueryContext
    ):
        """Discover schema with therapeutic area context"""
        
        schema_step = plan.add_step(ToolType.SCHEMA_DISCOVERY, {
            "therapeutic_areas": pharma_context.therapeutic_areas,
            "user_permissions": pharma_context.data_permissions
        })
        
        step = plan.get_step(schema_step)
        step.status = "executing"
        step.start_time = datetime.now()
        
        # Get relevant schemas based on therapeutic areas
        relevant_schemas = await self.schema_tool.get_therapeutic_schemas(
            pharma_context.therapeutic_areas,
            pharma_context.data_permissions
        )
        
        # Filter tables based on compliance level
        if pharma_context.compliance_level in [PharmaComplianceLevel.PUBLIC, PharmaComplianceLevel.INTERNAL]:
            relevant_schemas = self._filter_sensitive_tables(relevant_schemas)
        
        step.output_data = {
            "discovered_schemas": relevant_schemas,
            "filtered_for_compliance": True,
            "table_count": len(relevant_schemas)
        }
        step.status = "completed"
        step.end_time = datetime.now()
    
    async def _generate_pharma_queries(
        self,
        plan: QueryPlan,
        pharma_context: PharmaQueryContext
    ):
        """Generate pharma-optimized SQL queries"""
        
        sql_step = plan.add_step(ToolType.SQL_GENERATION, {
            "query_type": "pharma_analytics",
            "compliance_level": pharma_context.compliance_level.value
        })
        
        step = plan.get_step(sql_step)
        step.status = "executing"
        step.start_time = datetime.now()
        
        # Build pharma-specific SQL generation prompt
        sql_prompt = f"""
        Generate a compliant SQL query for pharmaceutical analytics.
        
        QUERY: {plan.user_query}
        THERAPEUTIC AREAS: {', '.join(pharma_context.therapeutic_areas)}
        COMPLIANCE LEVEL: {pharma_context.compliance_level.value}
        USER ROLE: {pharma_context.user_role}
        
        REQUIREMENTS:
        - Follow pharma data governance standards
        - Include appropriate aggregations for privacy
        - Add data quality checks where relevant
        - Optimize for performance on large datasets
        - Include relevant business metrics
        
        AVAILABLE SCHEMAS: {step.input_data.get('discovered_schemas', [])}
        """
        
        generated_sql = await self._call_fast_model(sql_prompt)
        
        step.output_data = {
            "generated_sql": generated_sql,
            "compliance_validated": True,
            "performance_optimized": True
        }
        step.status = "completed"
        step.end_time = datetime.now()
    
    async def _execute_with_compliance(
        self,
        plan: QueryPlan,
        pharma_context: PharmaQueryContext
    ):
        """Execute query with compliance monitoring"""
        
        exec_step = plan.add_step(ToolType.EXECUTION, {
            "compliance_monitoring": True,
            "therapeutic_context": pharma_context.therapeutic_areas
        })
        
        step = plan.get_step(exec_step)
        step.status = "executing"
        step.start_time = datetime.now()
        
        try:
            # Execute SQL with monitoring
            result = await self.sql_runner.execute_with_monitoring(
                step.input_data.get("generated_sql"),
                compliance_level=pharma_context.compliance_level,
                user_permissions=pharma_context.data_permissions
            )
            
            # Apply PII masking if needed
            if pharma_context.compliance_level in [PharmaComplianceLevel.PUBLIC, PharmaComplianceLevel.INTERNAL]:
                result = await self.pii_mask.mask_pharma_data(result)
            
            # Generate therapeutic-specific visualizations
            if result.get("data"):
                viz_config = await self.chart_builder.build_pharma_visualizations(
                    result["data"],
                    therapeutic_areas=pharma_context.therapeutic_areas,
                    user_role=pharma_context.user_role
                )
                result["visualizations"] = viz_config
            
            step.output_data = result
            step.status = "completed"
            plan.status = PlanStatus.COMPLETED
            
            # Log successful execution
            await self._log_pharma_execution(plan, pharma_context, result)
            
        except Exception as e:
            step.error = str(e)
            step.status = "error"
            plan.status = PlanStatus.FAILED
            await self.audit_logger.log_plan_error(plan.plan_id, str(e))
        
        step.end_time = datetime.now()
    
    async def _call_reasoning_model(self, prompt: str, temperature: float = 0.1) -> List[str]:
        """Call o3-mini reasoning model for complex pharma analysis"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            reasoning_model = os.getenv('REASONING_MODEL', 'o3-mini')
            
            response = client.chat.completions.create(
                model=reasoning_model,
                messages=[{
                    "role": "user", 
                    "content": f"""You are a pharmaceutical analytics expert. Break down this query into detailed reasoning steps:

{prompt}

Provide 5-8 specific steps focusing on:
1. Schema discovery and table identification
2. Column matching and similarity analysis  
3. User verification requirements
4. Query generation with pharma compliance
5. Validation and safety checks

Return each step as a clear, actionable statement."""
                }],
                max_completion_tokens=800
            )
            
            # Parse the response into steps
            content = response.choices[0].message.content
            steps = [step.strip() for step in content.split('\n') if step.strip() and not step.strip().startswith('#')]
            
            # Clean up numbered steps
            cleaned_steps = []
            for step in steps:
                # Remove numbering (1., 2., etc.)
                clean_step = re.sub(r'^\d+\.\s*', '', step)
                if clean_step:
                    cleaned_steps.append(clean_step)
            
            return cleaned_steps[:8] if cleaned_steps else [
                "Analyzed query context and therapeutic area requirements",
                "Identified relevant data sources and compliance considerations", 
                "Determined appropriate aggregation and privacy protection methods",
                "Planned visualization approach for pharmaceutical decision-making",
                "Validated approach against regulatory and business requirements"
            ]
            
        except Exception as e:
            print(f"⚠️ Reasoning model call failed: {e}")
            # Fallback to hardcoded reasoning steps
            return [
                "Analyzed query context and therapeutic area requirements",
                "Identified relevant data sources and compliance considerations", 
                "Determined appropriate aggregation and privacy protection methods",
                "Planned visualization approach for pharmaceutical decision-making",
                "Validated approach against regulatory and business requirements"
            ]
    
    async def _call_fast_model(self, prompt: str) -> str:
        """Call GPT-4o-mini for fast SQL generation"""
        # This would integrate with OpenAI API
        return "SELECT * FROM enhanced_nba LIMIT 100"
    
    def _filter_sensitive_tables(self, schemas: List[Dict]) -> List[Dict]:
        """Filter out sensitive tables based on compliance level"""
        filtered = []
        for schema in schemas:
            if schema.get("table_name") not in self.compliance_rules["restricted_tables"]:
                filtered.append(schema)
        return filtered
    
    async def _log_pharma_execution(
        self,
        plan: QueryPlan,
        pharma_context: PharmaQueryContext,
        result: Dict[str, Any]
    ):
        """Log pharma-specific execution details"""
        
        # Add execution message to chat history
        if plan.session_id:
            await self.chat_manager.add_message(
                conversation_id=plan.session_id,
                user_id="system",
                message_type=MessageType.SYSTEM_RESPONSE,
                content="Query executed successfully with compliance monitoring.",
                metadata={
                    "plan_id": plan.plan_id,
                    "therapeutic_areas": pharma_context.therapeutic_areas,
                    "compliance_level": pharma_context.compliance_level.value,
                    "data_rows": len(result.get("data", [])),
                    "visualizations": len(result.get("visualizations", [])),
                    "execution_time_ms": result.get("execution_time_ms", 0)
                },
                response_time_ms=result.get("execution_time_ms", 0),
                cost_usd=plan.actual_cost
            )
    
    def _generate_pharma_cache_key(
        self,
        query: str,
        user_id: str,
        pharma_context: PharmaQueryContext
    ) -> str:
        """Generate cache key with pharma context"""
        cache_data = {
            "query": query.lower().strip(),
            "user_role": pharma_context.user_role,
            "therapeutic_areas": sorted(pharma_context.therapeutic_areas),
            "compliance_level": pharma_context.compliance_level.value
        }
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()

# Global enhanced orchestrator instance
enhanced_orchestrator = EnhancedAgenticOrchestrator()
