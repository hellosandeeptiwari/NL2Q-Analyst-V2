"""
Intelligent Query Planner
Orchestrates database intelligence and query optimization
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import time

from .database_intelligence import (
    database_intelligence, 
    initialize_database_intelligence, 
    get_intelligent_query_plan,
    get_database_intelligence_summary,
    QueryPlan
)
from ..nl2sql.enhanced_generator import generate_sql_with_enhanced_schema
from ..nl2sql.guardrails import GuardrailConfig
from ..db.engine import get_adapter

@dataclass
class PlannerResponse:
    """Comprehensive response from intelligent planner"""
    success: bool
    sql: str
    explanation: str
    confidence: float
    performance_tier: str
    database_features: List[str]
    alternatives: List[str]
    warnings: List[str]
    execution_strategy: str
    metadata: Dict[str, Any]

class IntelligentQueryPlanner:
    """Agentic planner that orchestrates database understanding and query generation"""
    
    def __init__(self):
        self.intelligence_initialized = False
        self.last_connection_check = None
        self.database_profile = None
        
    def initialize(self, adapter=None) -> bool:
        """Initialize the planner with database intelligence"""
        print("🚀 Intelligent Query Planner: Initializing...")
        
        try:
            # Initialize database intelligence
            success = initialize_database_intelligence(adapter)
            
            if success:
                self.intelligence_initialized = True
                self.last_connection_check = time.time()
                self.database_profile = get_database_intelligence_summary()
                
                print("✅ Query Planner initialized successfully")
                print(f"   Database: {self.database_profile.get('engine')} {self.database_profile.get('version', '')}")
                print(f"   Tables: {self.database_profile.get('schema_tables', 0)}")
                print(f"   Features: {', '.join(self.database_profile.get('supported_features', [])[:3])}")
                
                return True
            else:
                print("❌ Database intelligence initialization failed")
                return False
                
        except Exception as e:
            print(f"❌ Planner initialization failed: {e}")
            return False
    
    def plan_and_execute_query(self, natural_language: str, context: Optional[Dict] = None) -> PlannerResponse:
        """Intelligently plan and generate optimized query"""
        
        if not self.intelligence_initialized:
            return PlannerResponse(
                success=False,
                sql="",
                explanation="Database intelligence not initialized",
                confidence=0.0,
                performance_tier="error",
                database_features=[],
                alternatives=[],
                warnings=["Initialize planner before use"],
                execution_strategy="none",
                metadata={"error": "not_initialized"}
            )
        
        print(f"🧠 Planning intelligent query for: {natural_language}")
        
        try:
            # Step 1: Get intelligent query plan
            query_plan = get_intelligent_query_plan(natural_language, context)
            
            # Step 2: Enhance with advanced SQL generation if needed
            enhanced_sql = self._enhance_with_llm_generation(natural_language, query_plan, context)
            
            # Step 3: Validate and optimize
            final_sql, validation_results = self._validate_and_optimize(enhanced_sql, query_plan)
            
            # Step 4: Determine execution strategy
            execution_strategy = self._determine_execution_strategy(final_sql, validation_results)
            
            # Step 5: Calculate confidence score
            confidence = self._calculate_confidence(query_plan, validation_results)
            
            return PlannerResponse(
                success=True,
                sql=final_sql,
                explanation=f"Intelligently generated for {self.database_profile['engine']}: {query_plan.explanation}",
                confidence=confidence,
                performance_tier=query_plan.performance_tier,
                database_features=query_plan.database_features_used,
                alternatives=[alt.get("description", "") for alt in query_plan.alternatives],
                warnings=query_plan.warnings + validation_results.get("warnings", []),
                execution_strategy=execution_strategy,
                metadata={
                    "database_engine": self.database_profile.get("engine"),
                    "query_complexity": validation_results.get("complexity", "medium"),
                    "optimization_applied": validation_results.get("optimizations", []),
                    "estimated_cost": query_plan.estimated_cost
                }
            )
            
        except Exception as e:
            print(f"❌ Query planning failed: {e}")
            return PlannerResponse(
                success=False,
                sql="SELECT 'Query planning failed' as error",
                explanation=f"Planning error: {str(e)}",
                confidence=0.0,
                performance_tier="error",
                database_features=[],
                alternatives=[],
                warnings=[f"Planning failed: {str(e)}"],
                execution_strategy="fallback",
                metadata={"error": str(e)}
            )
    
    def _enhance_with_llm_generation(self, natural_language: str, query_plan: QueryPlan, context: Optional[Dict]) -> str:
        """Enhance basic query plan with LLM-powered generation"""
        
        # Check if we need LLM enhancement (complex queries)
        if query_plan.performance_tier in ["poor", "acceptable"] or len(query_plan.warnings) > 2:
            print("  🤖 Enhancing with LLM generation...")
            
            try:
                # Get enhanced schema
                enhanced_schema = database_intelligence.schema_snapshot
                
                if enhanced_schema and "schema_version" in enhanced_schema:
                    # Use enhanced generator with database intelligence
                    guardrails = GuardrailConfig(
                        enable_write=False,
                        allowed_schemas=["public"],
                        default_limit=100
                    )
                    
                    generated = generate_sql_with_enhanced_schema(
                        natural_language, 
                        enhanced_schema, 
                        guardrails
                    )
                    
                    # Choose best between agent plan and LLM generation
                    if len(generated.sql) > len(query_plan.sql) and "SELECT" in generated.sql.upper():
                        print("  ✅ LLM-enhanced query selected")
                        return generated.sql
                
            except Exception as e:
                print(f"  ⚠️ LLM enhancement failed: {e}")
        
        # Use agent-generated query
        return query_plan.sql
    
    def _validate_and_optimize(self, sql: str, query_plan: QueryPlan) -> tuple[str, Dict[str, Any]]:
        """Validate and apply database-specific optimizations"""
        
        validation_results = {
            "complexity": "medium",
            "optimizations": [],
            "warnings": []
        }
        
        optimized_sql = sql
        
        # Apply database-specific optimizations
        if self.database_profile.get("engine") == "sqlite":
            # SQLite optimizations
            if "LIMIT" not in optimized_sql.upper():
                optimized_sql += " LIMIT 1000"
                validation_results["optimizations"].append("Added LIMIT clause")
                
        elif self.database_profile.get("engine") == "postgresql":
            # PostgreSQL optimizations
            if "EXPLAIN" not in optimized_sql.upper() and len(optimized_sql) > 200:
                validation_results["warnings"].append("Consider using EXPLAIN ANALYZE for complex queries")
                
        elif self.database_profile.get("engine") == "snowflake":
            # Snowflake optimizations
            if "LIMIT" not in optimized_sql.upper():
                optimized_sql += " LIMIT 10000"
                validation_results["optimizations"].append("Added Snowflake-appropriate LIMIT")
        
        # Assess complexity
        complexity_score = sum([
            "JOIN" in optimized_sql.upper(),
            "GROUP BY" in optimized_sql.upper(), 
            "ORDER BY" in optimized_sql.upper(),
            "HAVING" in optimized_sql.upper(),
            "UNION" in optimized_sql.upper()
        ])
        
        if complexity_score <= 1:
            validation_results["complexity"] = "low"
        elif complexity_score >= 3:
            validation_results["complexity"] = "high"
        
        return optimized_sql, validation_results
    
    def _determine_execution_strategy(self, sql: str, validation_results: Dict) -> str:
        """Determine the best execution strategy"""
        
        if validation_results["complexity"] == "high":
            return "careful_execution_with_monitoring"
        elif len(validation_results.get("warnings", [])) > 0:
            return "execute_with_caution"
        else:
            return "direct_execution"
    
    def _calculate_confidence(self, query_plan: QueryPlan, validation_results: Dict) -> float:
        """Calculate confidence score for the query"""
        
        base_confidence = 0.7
        
        # Boost confidence for optimal performance
        if query_plan.performance_tier == "optimal":
            base_confidence += 0.2
        elif query_plan.performance_tier == "good":
            base_confidence += 0.1
        elif query_plan.performance_tier == "poor":
            base_confidence -= 0.2
        
        # Reduce confidence for warnings
        warning_penalty = len(query_plan.warnings) * 0.05
        base_confidence -= warning_penalty
        
        # Boost confidence for applied optimizations
        optimization_boost = len(validation_results.get("optimizations", [])) * 0.05
        base_confidence += optimization_boost
        
        return max(0.0, min(1.0, base_confidence))
    
    def get_planner_status(self) -> Dict[str, Any]:
        """Get comprehensive planner status"""
        
        return {
            "initialized": self.intelligence_initialized,
            "database_profile": self.database_profile,
            "last_connection_check": self.last_connection_check,
            "capabilities": {
                "intelligent_planning": self.intelligence_initialized,
                "llm_enhancement": True,
                "database_optimization": True,
                "performance_assessment": True,
                "multi_strategy_execution": True
            },
            "supported_databases": ["SQLite", "PostgreSQL", "Snowflake", "MySQL"],
            "optimization_features": [
                "Database-specific syntax optimization",
                "Performance tier assessment", 
                "Intelligent LIMIT clause insertion",
                "Query complexity analysis",
                "Alternative strategy generation"
            ]
        }
    
    def suggest_query_improvements(self, sql: str) -> List[str]:
        """Suggest improvements for existing queries"""
        
        if not self.intelligence_initialized:
            return ["Initialize planner first"]
        
        suggestions = []
        sql_upper = sql.upper()
        
        # Database-specific suggestions
        engine = self.database_profile.get("engine", "").lower()
        
        if "LIMIT" not in sql_upper:
            suggestions.append(f"Add LIMIT clause to prevent large result sets")
        
        if "INDEX" not in sql_upper and "JOIN" in sql_upper:
            suggestions.append("Consider creating indexes on JOIN columns")
        
        if engine == "postgresql" and "EXPLAIN" not in sql_upper:
            suggestions.append("Use EXPLAIN ANALYZE to optimize query performance")
        
        if engine == "snowflake" and "CLUSTER" not in sql_upper:
            suggestions.append("Consider clustering keys for large table queries")
        
        if len(sql) > 500:
            suggestions.append("Consider breaking complex query into simpler parts")
        
        return suggestions

# Global planner instance
intelligent_planner = IntelligentQueryPlanner()

def initialize_intelligent_planner(adapter=None) -> bool:
    """Initialize the intelligent query planner"""
    return intelligent_planner.initialize(adapter)

def get_intelligent_query_response(natural_language: str, context: Optional[Dict] = None) -> PlannerResponse:
    """Get intelligent query response from planner"""
    return intelligent_planner.plan_and_execute_query(natural_language, context)

def get_planner_status() -> Dict[str, Any]:
    """Get planner status"""
    return intelligent_planner.get_planner_status()
