"""
Agentic Database Documentation System
Dynamically analyzes database capabilities and provides intelligent query planning
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import time
from datetime import datetime

from ..db.database_documentation import get_database_documentation, DatabaseDialect
from ..db.engine import get_adapter
from ..db.enhanced_schema import EnhancedSchemaBuilder

@dataclass
class DatabaseCapabilityProfile:
    """Complete database capability profile for intelligent query planning"""
    engine_name: str
    version: Optional[str]
    capabilities: Dict[str, Any]
    syntax_rules: Dict[str, str]
    performance_hints: List[str]
    limitations: List[str]
    function_catalog: Dict[str, Dict[str, str]]
    query_patterns: Dict[str, str]
    optimization_strategies: List[str]
    compliance_rules: List[str]

@dataclass
class QueryPlan:
    """Intelligent query plan with database-specific optimizations"""
    sql: str
    explanation: str
    performance_tier: str  # "optimal", "good", "acceptable", "poor"
    estimated_cost: str
    alternatives: List[Dict[str, str]]
    warnings: List[str]
    database_features_used: List[str]

class DatabaseIntelligenceAgent:
    """Agentic system for understanding and optimizing database interactions"""
    
    def __init__(self):
        self.adapter = None
        self.capability_profile = None
        self.schema_snapshot = None
        self.connection_established = False
        
    def establish_database_connection(self, adapter=None) -> bool:
        """Establish connection and analyze database capabilities"""
        print("🔗 Database Intelligence Agent: Establishing connection...")
        
        try:
            self.adapter = adapter or get_adapter()
            health = self.adapter.health()
            
            if not health.get("connected"):
                print("❌ Database connection failed")
                return False
                
            print("✅ Database connection established")
            self.connection_established = True
            
            # Analyze database capabilities
            self._analyze_database_capabilities()
            
            # Build enhanced schema understanding
            self._build_schema_intelligence()
            
            print("🧠 Database intelligence analysis complete")
            return True
            
        except Exception as e:
            print(f"❌ Database intelligence setup failed: {e}")
            return False
    
    def _analyze_database_capabilities(self):
        """Deep analysis of database capabilities and features"""
        print("  🔍 Analyzing database capabilities...")
        
        # Detect engine type and version
        engine_type = self._detect_engine_type()
        version = self._detect_version()
        
        # Get base documentation
        dialect = get_database_documentation(engine_type, version)
        
        # Perform capability tests
        capabilities = self._test_database_capabilities()
        
        # Build complete capability profile
        self.capability_profile = DatabaseCapabilityProfile(
            engine_name=engine_type,
            version=version,
            capabilities=capabilities,
            syntax_rules=self._extract_syntax_rules(dialect),
            performance_hints=dialect.performance_tips,
            limitations=dialect.limitations,
            function_catalog=self._build_function_catalog(dialect),
            query_patterns=dialect.common_patterns,
            optimization_strategies=self._generate_optimization_strategies(),
            compliance_rules=self._generate_compliance_rules()
        )
        
        print(f"  ✅ Capability profile created for {engine_type} {version or ''}")
    
    def _build_schema_intelligence(self):
        """Build intelligent schema understanding"""
        print("  📊 Building schema intelligence...")
        
        try:
            builder = EnhancedSchemaBuilder()
            self.schema_snapshot = builder.build_enhanced_snapshot(self.adapter)
            print(f"  ✅ Schema analyzed: {len(self.schema_snapshot.get('tables', []))} tables")
        except Exception as e:
            print(f"  ⚠️ Schema analysis limited: {e}")
            self.schema_snapshot = {"tables": [], "basic_mode": True}
    
    def _detect_engine_type(self) -> str:
        """Detect database engine type through introspection"""
        try:
            # Try engine-specific queries
            if hasattr(self.adapter, 'conn'):
                if 'sqlite' in str(type(self.adapter.conn)).lower():
                    return "sqlite"
                elif 'psycopg' in str(type(self.adapter.conn)).lower():
                    return "postgresql"
                elif 'snowflake' in str(type(self.adapter.conn)).lower():
                    return "snowflake"
                elif 'mysql' in str(type(self.adapter.conn)).lower():
                    return "mysql"
            
            # Fallback to SQL-based detection
            try:
                cur = self.adapter.conn.cursor()
                cur.execute("SELECT sqlite_version()")
                cur.close()
                return "sqlite"
            except:
                pass
                
            return "unknown"
        except Exception:
            return "unknown"
    
    def _detect_version(self) -> Optional[str]:
        """Detect database version"""
        try:
            engine = self.capability_profile.engine_name if self.capability_profile else self._detect_engine_type()
            
            if engine == "sqlite":
                cur = self.adapter.conn.cursor()
                cur.execute("SELECT sqlite_version()")
                version = cur.fetchone()[0]
                cur.close()
                return version
            elif engine == "postgresql":
                cur = self.adapter.conn.cursor()
                cur.execute("SELECT version()")
                version_info = cur.fetchone()[0]
                cur.close()
                return version_info.split()[1] if version_info else None
            # Add more database version detection as needed
            
        except Exception:
            pass
        return None
    
    def _test_database_capabilities(self) -> Dict[str, Any]:
        """Test specific database capabilities"""
        capabilities = {
            "supports_cte": False,
            "supports_window_functions": False,
            "supports_json": False,
            "supports_arrays": False,
            "supports_recursive": False,
            "max_query_complexity": "medium",
            "indexing_capabilities": [],
            "transaction_isolation": "read_committed"
        }
        
        # Test CTE support
        try:
            cur = self.adapter.conn.cursor()
            cur.execute("WITH test AS (SELECT 1 as n) SELECT n FROM test")
            cur.close()
            capabilities["supports_cte"] = True
        except:
            pass
        
        # Test window function support
        try:
            cur = self.adapter.conn.cursor()
            cur.execute("SELECT ROW_NUMBER() OVER (ORDER BY 1) FROM (SELECT 1) t")
            cur.close()
            capabilities["supports_window_functions"] = True
        except:
            pass
        
        # Test JSON support (database-specific)
        engine = self._detect_engine_type()
        if engine in ["postgresql", "mysql", "sqlite"]:
            capabilities["supports_json"] = True
        
        return capabilities
    
    def _extract_syntax_rules(self, dialect: DatabaseDialect) -> Dict[str, str]:
        """Extract key syntax rules for query generation"""
        return {
            "quote_char": dialect.quote_char,
            "limit_syntax": dialect.limit_syntax,
            "case_sensitivity": dialect.case_sensitivity,
            "string_concat": dialect.string_functions.get("concat", "CONCAT(str1, str2)"),
            "current_date": dialect.date_functions.get("current_date", "CURRENT_DATE"),
            "current_timestamp": dialect.date_functions.get("current_timestamp", "CURRENT_TIMESTAMP")
        }
    
    def _build_function_catalog(self, dialect: DatabaseDialect) -> Dict[str, Dict[str, str]]:
        """Build comprehensive function catalog"""
        return {
            "date_functions": dialect.date_functions,
            "string_functions": dialect.string_functions,
            "aggregate_functions": {func: f"{func}(column)" for func in dialect.aggregate_functions},
            "window_functions": {func: f"{func}() OVER (...)" for func in dialect.window_functions}
        }
    
    def _generate_optimization_strategies(self) -> List[str]:
        """Generate database-specific optimization strategies"""
        engine = self._detect_engine_type()
        
        if engine == "sqlite":
            return [
                "Create indexes on frequently queried columns",
                "Use LIMIT to prevent large result sets",
                "Avoid complex JOINs with large tables",
                "Use EXPLAIN QUERY PLAN for optimization"
            ]
        elif engine == "postgresql":
            return [
                "Use EXPLAIN ANALYZE for query planning",
                "Create partial indexes for filtered queries",
                "Use appropriate JOIN algorithms",
                "Consider table partitioning for large datasets"
            ]
        elif engine == "snowflake":
            return [
                "Use clustering keys for large tables",
                "Leverage automatic query optimization",
                "Consider result caching for repeated queries",
                "Use appropriate warehouse sizes"
            ]
        else:
            return [
                "Create indexes on frequently queried columns",
                "Use LIMIT clauses for large result sets",
                "Optimize WHERE clauses for performance"
            ]
    
    def _generate_compliance_rules(self) -> List[str]:
        """Generate compliance and governance rules"""
        return [
            "Always use parameterized queries to prevent SQL injection",
            "Respect row limits to prevent resource exhaustion", 
            "Apply data governance rules for sensitive data",
            "Use appropriate aggregation for privacy protection",
            "Follow least-privilege access patterns"
        ]
    
    def plan_query(self, natural_language: str, context: Optional[Dict] = None) -> QueryPlan:
        """Intelligently plan and optimize a query based on database capabilities"""
        
        if not self.connection_established:
            raise Exception("Database connection not established. Call establish_database_connection() first.")
        
        print(f"🎯 Planning query: {natural_language}")
        
        # Analyze query requirements
        query_analysis = self._analyze_query_requirements(natural_language)
        
        # Generate optimized SQL
        optimized_sql = self._generate_optimized_sql(natural_language, query_analysis)
        
        # Assess performance
        performance_assessment = self._assess_query_performance(optimized_sql)
        
        # Generate alternatives
        alternatives = self._generate_query_alternatives(natural_language, query_analysis)
        
        # Check for warnings
        warnings = self._check_query_warnings(optimized_sql, query_analysis)
        
        # Identify database features used
        features_used = self._identify_features_used(optimized_sql)
        
        return QueryPlan(
            sql=optimized_sql,
            explanation=f"Optimized for {self.capability_profile.engine_name} using {', '.join(features_used)}",
            performance_tier=performance_assessment["tier"],
            estimated_cost=performance_assessment["cost"],
            alternatives=alternatives,
            warnings=warnings,
            database_features_used=features_used
        )
    
    def _analyze_query_requirements(self, natural_language: str) -> Dict[str, Any]:
        """Analyze what the query needs to accomplish"""
        # Simple analysis - in production this could use NLP
        analysis = {
            "requires_aggregation": any(word in natural_language.lower() for word in ["count", "sum", "avg", "max", "min", "total"]),
            "requires_filtering": any(word in natural_language.lower() for word in ["where", "filter", "only", "specific"]),
            "requires_sorting": any(word in natural_language.lower() for word in ["top", "bottom", "order", "sort", "highest", "lowest"]),
            "requires_grouping": any(word in natural_language.lower() for word in ["by", "group", "each", "per"]),
            "requires_joins": any(word in natural_language.lower() for word in ["and", "with", "from", "across"]),
            "estimated_complexity": "medium"
        }
        
        # Assess complexity
        complexity_indicators = sum([
            analysis["requires_aggregation"],
            analysis["requires_filtering"], 
            analysis["requires_sorting"],
            analysis["requires_grouping"],
            analysis["requires_joins"]
        ])
        
        if complexity_indicators <= 1:
            analysis["estimated_complexity"] = "low"
        elif complexity_indicators >= 4:
            analysis["estimated_complexity"] = "high"
            
        return analysis
    
    def _generate_optimized_sql(self, natural_language: str, analysis: Dict) -> str:
        """Generate database-optimized SQL"""
        # This would integrate with the enhanced SQL generator
        # For now, return a placeholder that demonstrates the concept
        
        tables = self.schema_snapshot.get("tables", [])
        if not tables:
            return "SELECT 'No tables available' as message"
        
        # Simple example generation based on capabilities
        main_table = tables[0]["name"]
        columns = [col["name"] for col in tables[0].get("columns", [])]
        
        sql_parts = []
        
        # SELECT clause
        if analysis["requires_aggregation"]:
            sql_parts.append(f"SELECT COUNT(*) as total_count FROM {main_table}")
        else:
            sql_parts.append(f"SELECT {', '.join(columns[:3])} FROM {main_table}")
        
        # Add LIMIT based on database capabilities
        limit_syntax = self.capability_profile.syntax_rules.get("limit_syntax", "LIMIT 100")
        if "LIMIT" in limit_syntax:
            sql_parts.append("LIMIT 100")
        
        return " ".join(sql_parts)
    
    def _assess_query_performance(self, sql: str) -> Dict[str, str]:
        """Assess query performance characteristics"""
        # Simple assessment based on query structure
        if "JOIN" in sql.upper():
            return {"tier": "good", "cost": "medium"}
        elif "COUNT(*)" in sql.upper():
            return {"tier": "optimal", "cost": "low"}
        else:
            return {"tier": "good", "cost": "low"}
    
    def _generate_query_alternatives(self, natural_language: str, analysis: Dict) -> List[Dict[str, str]]:
        """Generate alternative query approaches"""
        alternatives = []
        
        if analysis["requires_aggregation"]:
            alternatives.append({
                "approach": "Aggregated view",
                "description": "Use pre-computed aggregations if available"
            })
        
        if analysis["estimated_complexity"] == "high":
            alternatives.append({
                "approach": "Simplified query",
                "description": "Break into multiple simpler queries"
            })
        
        return alternatives
    
    def _check_query_warnings(self, sql: str, analysis: Dict) -> List[str]:
        """Check for potential query issues"""
        warnings = []
        
        if "LIMIT" not in sql.upper():
            warnings.append("Consider adding LIMIT clause to prevent large result sets")
        
        if analysis["estimated_complexity"] == "high":
            warnings.append("Complex query detected - monitor performance")
        
        if not self.capability_profile.capabilities.get("supports_window_functions") and "OVER" in sql.upper():
            warnings.append("Window functions not supported in this database version")
        
        return warnings
    
    def _identify_features_used(self, sql: str) -> List[str]:
        """Identify database features used in the query"""
        features = []
        
        if "JOIN" in sql.upper():
            features.append("Table Joins")
        if "GROUP BY" in sql.upper():
            features.append("Grouping")
        if "ORDER BY" in sql.upper():
            features.append("Sorting")
        if "LIMIT" in sql.upper():
            features.append("Result Limiting")
        if any(func in sql.upper() for func in ["COUNT", "SUM", "AVG", "MAX", "MIN"]):
            features.append("Aggregation Functions")
            
        return features or ["Basic Selection"]
    
    def get_database_summary(self) -> Dict[str, Any]:
        """Get comprehensive database capability summary"""
        if not self.connection_established:
            return {"status": "not_connected"}
        
        return {
            "status": "connected",
            "engine": self.capability_profile.engine_name,
            "version": self.capability_profile.version,
            "capabilities": self.capability_profile.capabilities,
            "schema_tables": len(self.schema_snapshot.get("tables", [])),
            "optimization_strategies": len(self.capability_profile.optimization_strategies),
            "supported_features": [
                feature for feature, supported in self.capability_profile.capabilities.items() 
                if supported and isinstance(supported, bool)
            ],
            "performance_recommendations": self.capability_profile.performance_hints[:3]
        }

# Global instance for the planner to use
database_intelligence = DatabaseIntelligenceAgent()

def initialize_database_intelligence(adapter=None) -> bool:
    """Initialize the database intelligence system"""
    return database_intelligence.establish_database_connection(adapter)

def get_intelligent_query_plan(natural_language: str, context: Optional[Dict] = None) -> QueryPlan:
    """Get an intelligent query plan from the agent"""
    return database_intelligence.plan_query(natural_language, context)

def get_database_intelligence_summary() -> Dict[str, Any]:
    """Get database intelligence summary"""
    return database_intelligence.get_database_summary()
