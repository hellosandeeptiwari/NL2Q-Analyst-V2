"""
Schema Semantic Analyzer
========================

This module handles deep semantic analysis of database schemas to provide
intelligent context for query planning and generation.

Key Features:
- Extract business meaning from table and column names
- Identify relationships and patterns
- Provide semantic context for LLM query planning
- Support multiple business domains (Healthcare, Finance, etc.)
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class SchemaSemanticAnalyzer:
    """
    Analyzes database schemas to extract semantic meaning and business context
    for intelligent query planning.
    """
    
    def __init__(self):
        self.logger = logger
        self.business_domain_patterns = self._load_business_domain_patterns()
        self.column_type_patterns = self._load_column_type_patterns()
        
    def _load_business_domain_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Load business domain patterns for semantic analysis"""
        return {
            "healthcare_pharma": {
                "table_indicators": [
                    "prescriber", "patient", "physician", "doctor", "medical", "clinical",
                    "drug", "medication", "pharma", "therapeutic", "treatment",
                    "nrx", "trx", "prescription", "rx", "dose", "therapy"
                ],
                "column_indicators": [
                    "prescriber", "physician", "doctor", "patient", "npi", "dea",
                    "specialty", "therapeutic", "indication", "dosage",
                    "trx", "nrx", "tqty", "nqty", "units", "vials"
                ],
                "metric_patterns": [
                    "trx", "nrx", "tqty", "nqty", "market_share", "volume",
                    "units", "calls", "samples", "lunch_learn"
                ]
            },
            "sales_marketing": {
                "table_indicators": [
                    "territory", "region", "area", "zone", "rep", "sales",
                    "target", "quota", "performance", "achievement"
                ],
                "column_indicators": [
                    "territory", "region", "rep", "sales", "target", "quota",
                    "achievement", "performance", "tier", "ranking"
                ],
                "metric_patterns": [
                    "target", "actual", "achievement", "quota", "performance",
                    "ranking", "tier", "score", "rate"
                ]
            },
            "financial": {
                "table_indicators": [
                    "payment", "invoice", "billing", "revenue", "cost",
                    "price", "rebate", "discount", "fee"
                ],
                "column_indicators": [
                    "amount", "price", "cost", "revenue", "fee", "rebate",
                    "discount", "payment", "billing", "invoice"
                ],
                "metric_patterns": [
                    "amount", "total", "sum", "cost", "price", "fee",
                    "revenue", "profit", "margin"
                ]
            },
            "geographical": {
                "table_indicators": [
                    "location", "address", "geography", "geo", "spatial"
                ],
                "column_indicators": [
                    "address", "city", "state", "zip", "zipcode", "country",
                    "latitude", "longitude", "coordinates", "location"
                ],
                "metric_patterns": [
                    "distance", "radius", "area", "coverage", "density"
                ]
            }
        }
    
    def _load_column_type_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for identifying column types"""
        return {
            "identifier": [
                "id", "key", "code", "number", "num", "ref", "reference",
                "pk", "fk", "uuid", "guid"
            ],
            "temporal": [
                "date", "time", "timestamp", "created", "updated", "modified",
                "start", "end", "begin", "finish", "period", "year", "month",
                "day", "week", "quarter", "qtr"
            ],
            "categorical": [
                "type", "category", "class", "group", "status", "state",
                "flag", "tier", "level", "grade", "rank", "priority",
                "name", "title", "description", "label"
            ],
            "metric": [
                "count", "total", "sum", "avg", "average", "min", "max",
                "rate", "ratio", "percent", "share", "volume", "quantity",
                "qty", "units", "amount", "value", "score"
            ],
            "geographical": [
                "address", "city", "state", "zip", "zipcode", "country",
                "region", "territory", "area", "zone", "location", "lat", "lng"
            ],
            "boolean": [
                "flag", "is_", "has_", "can_", "active", "enabled", "valid"
            ]
        }
    
    async def analyze_schema_semantics(
        self, 
        table_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive semantic analysis of schema metadata.
        
        Args:
            table_metadata: Dictionary containing table names and their column information
            
        Returns:
            Comprehensive semantic analysis results
        """
        try:
            self.logger.info(f"🧠 Analyzing schema semantics for {len(table_metadata)} tables")
            
            analysis = {
                "tables": {},
                "cross_table_relationships": {},
                "business_domains": {},
                "query_capabilities": {},
                "semantic_summary": {}
            }
            
            # Analyze each table individually
            for table_name, metadata in table_metadata.items():
                analysis["tables"][table_name] = await self._analyze_table_semantics(
                    table_name, metadata
                )
            
            # Analyze relationships between tables
            analysis["cross_table_relationships"] = self._analyze_cross_table_relationships(
                analysis["tables"]
            )
            
            # Identify business domains
            analysis["business_domains"] = self._identify_business_domains(
                analysis["tables"]
            )
            
            # Determine query capabilities
            analysis["query_capabilities"] = self._determine_query_capabilities(
                analysis["tables"], analysis["cross_table_relationships"]
            )
            
            # Create semantic summary
            analysis["semantic_summary"] = self._create_semantic_summary(analysis)
            
            self.logger.info("✅ Schema semantic analysis completed")
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error in schema semantic analysis: {e}")
            return {}
    
    async def _analyze_table_semantics(
        self, 
        table_name: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze semantics of individual table"""
        
        table_analysis = {
            "table_name": table_name,
            "business_purpose": "",
            "data_categories": {},
            "column_semantics": {},
            "relationship_potential": {},
            "query_patterns": [],
            "complexity_indicators": {}
        }
        
        columns = metadata.get("columns", [])
        
        # Infer business purpose from table name
        table_analysis["business_purpose"] = self._infer_table_business_purpose(table_name)
        
        # Categorize columns by semantic type
        table_analysis["data_categories"] = self._categorize_columns_semantically(columns)
        
        # Analyze individual column semantics
        for column in columns:
            table_analysis["column_semantics"][column] = self._analyze_column_semantics(
                column, table_name
            )
        
        # Assess relationship potential
        table_analysis["relationship_potential"] = self._assess_relationship_potential(
            table_name, columns
        )
        
        # Identify supported query patterns
        table_analysis["query_patterns"] = self._identify_query_patterns(
            table_name, table_analysis["data_categories"]
        )
        
        # Assess complexity indicators
        table_analysis["complexity_indicators"] = self._assess_table_complexity(
            table_name, columns, table_analysis["data_categories"]
        )
        
        return table_analysis
    
    def _infer_table_business_purpose(self, table_name: str) -> str:
        """Infer business purpose from table name"""
        table_lower = table_name.lower()
        
        # Healthcare/Pharma patterns
        if any(indicator in table_lower for indicator in 
               self.business_domain_patterns["healthcare_pharma"]["table_indicators"]):
            if "prescriber" in table_lower:
                return "Healthcare provider/prescriber analytics and tracking"
            elif "patient" in table_lower:
                return "Patient demographics and treatment tracking"
            elif any(word in table_lower for word in ["trx", "nrx", "prescription"]):
                return "Prescription transaction and volume analysis"
            else:
                return "Healthcare/pharmaceutical data management"
        
        # Sales/Marketing patterns
        elif any(indicator in table_lower for indicator in 
                self.business_domain_patterns["sales_marketing"]["table_indicators"]):
            if "territory" in table_lower or "region" in table_lower:
                return "Sales territory and regional performance management"
            elif "target" in table_lower:
                return "Sales targeting and goal management"
            else:
                return "Sales and marketing performance tracking"
        
        # Financial patterns
        elif any(indicator in table_lower for indicator in 
                self.business_domain_patterns["financial"]["table_indicators"]):
            return "Financial transaction and payment processing"
        
        # Reporting/BI patterns
        elif any(word in table_lower for word in ["reporting", "bi", "overview", "summary"]):
            return "Business intelligence and reporting aggregation"
        
        else:
            return "General business data management"
    
    def _categorize_columns_semantically(self, columns: List[str]) -> Dict[str, List[str]]:
        """Categorize columns by semantic type"""
        categories = {
            "identifiers": [],
            "temporal": [],
            "categorical": [],
            "metrics": [],
            "geographical": [],
            "boolean": [],
            "descriptive": [],
            "relationship": []
        }
        
        for column in columns:
            column_lower = column.lower()
            categorized = False
            
            # Check each pattern type
            for category, patterns in self.column_type_patterns.items():
                if any(pattern in column_lower for pattern in patterns):
                    categories[category].append(column)
                    categorized = True
                    break
            
            # Special case for relationship columns (foreign keys)
            if any(word in column_lower for word in ["id", "key"]) and "_" in column_lower:
                categories["relationship"].append(column)
                categorized = True
            
            # If not categorized, add to descriptive
            if not categorized:
                categories["descriptive"].append(column)
        
        return categories
    
    def _analyze_column_semantics(self, column: str, table_name: str) -> Dict[str, Any]:
        """Analyze semantic meaning of individual column"""
        column_lower = column.lower()
        table_lower = table_name.lower()
        
        semantics = {
            "column_name": column,
            "semantic_type": "unknown",
            "business_meaning": "",
            "data_operations": [],
            "relationship_hints": [],
            "query_relevance": 0.5  # Default relevance
        }
        
        # Determine semantic type
        for category, patterns in self.column_type_patterns.items():
            if any(pattern in column_lower for pattern in patterns):
                semantics["semantic_type"] = category
                break
        
        # Infer business meaning
        semantics["business_meaning"] = self._infer_column_business_meaning(
            column, table_name, semantics["semantic_type"]
        )
        
        # Determine suitable data operations
        semantics["data_operations"] = self._determine_column_operations(
            semantics["semantic_type"], column_lower
        )
        
        # Identify relationship hints
        if "id" in column_lower and column_lower != "id":
            semantics["relationship_hints"].append(f"potential_fk_to_{column_lower.replace('id', '').replace('_', '')}")
        
        # Assess query relevance
        semantics["query_relevance"] = self._assess_column_query_relevance(
            column, semantics["semantic_type"], table_name
        )
        
        return semantics
    
    def _infer_column_business_meaning(
        self, 
        column: str, 
        table_name: str, 
        semantic_type: str
    ) -> str:
        """Infer business meaning of column"""
        column_lower = column.lower()
        table_lower = table_name.lower()
        
        # Healthcare/Pharma specific meanings
        if "prescriber" in table_lower:
            if "trx" in column_lower:
                return "Prescription transaction volume/count"
            elif "nrx" in column_lower:
                return "New prescription volume/count"
            elif "specialty" in column_lower:
                return "Healthcare provider specialty classification"
            elif "target" in column_lower:
                return "Sales targeting tier or priority"
        
        # Territory/Regional meanings
        if any(word in table_lower for word in ["territory", "region"]):
            if "name" in column_lower:
                return "Geographic or organizational unit identifier"
            elif "id" in column_lower:
                return "Unique identifier for geographic/organizational unit"
        
        # Generic meanings based on semantic type
        if semantic_type == "metric":
            return f"Quantitative measure or calculated value"
        elif semantic_type == "categorical":
            return f"Classification or grouping attribute"
        elif semantic_type == "temporal":
            return f"Time-based tracking or timestamp"
        elif semantic_type == "identifier":
            return f"Unique identifier or reference key"
        else:
            return f"Column {column}"
    
    def _determine_column_operations(self, semantic_type: str, column_lower: str) -> List[str]:
        """Determine suitable operations for column type"""
        operations = ["select", "filter"]  # Basic operations for all columns
        
        if semantic_type == "metric":
            operations.extend(["aggregate", "sum", "count", "average", "group_by"])
        elif semantic_type == "categorical":
            operations.extend(["group_by", "count", "distinct"])
        elif semantic_type == "temporal":
            operations.extend(["order_by", "filter", "group_by", "date_functions"])
        elif semantic_type == "identifier":
            operations.extend(["group_by", "join", "distinct", "count"])
        elif semantic_type == "geographical":
            operations.extend(["group_by", "spatial_functions", "hierarchy"])
        
        return operations
    
    def _assess_column_query_relevance(
        self, 
        column: str, 
        semantic_type: str, 
        table_name: str
    ) -> float:
        """Assess how relevant a column is for typical queries"""
        relevance = 0.5  # Base relevance
        
        column_lower = column.lower()
        
        # Higher relevance for commonly queried column types
        if semantic_type == "metric":
            relevance += 0.3
        elif semantic_type == "categorical":
            relevance += 0.2
        elif semantic_type == "temporal":
            relevance += 0.2
        elif semantic_type == "identifier" and "id" not in column_lower:
            relevance += 0.1
        
        # Healthcare/pharma specific relevance boosts
        if any(word in column_lower for word in ["trx", "nrx", "target", "specialty"]):
            relevance += 0.2
        
        # Territory/regional relevance boosts  
        if any(word in column_lower for word in ["territory", "region", "area"]):
            relevance += 0.15
        
        return min(relevance, 1.0)  # Cap at 1.0
    
    def _assess_relationship_potential(
        self, 
        table_name: str, 
        columns: List[str]
    ) -> Dict[str, Any]:
        """Assess potential for relationships with other tables"""
        potential = {
            "foreign_key_candidates": [],
            "join_columns": [],
            "hierarchy_indicators": [],
            "bridge_table_potential": False
        }
        
        for column in columns:
            column_lower = column.lower()
            
            # Foreign key candidates
            if "id" in column_lower and column_lower != "id":
                potential["foreign_key_candidates"].append({
                    "column": column,
                    "target_table_hint": column_lower.replace("id", "").replace("_", ""),
                    "confidence": 0.8
                })
            
            # Common join columns
            if any(word in column_lower for word in 
                   ["territory", "region", "area", "prescriber", "patient", "product"]):
                potential["join_columns"].append(column)
        
        # Hierarchy indicators
        if any(word in table_name.lower() for word in ["territory", "region", "area"]):
            potential["hierarchy_indicators"] = ["geographical_hierarchy"]
        
        # Bridge table potential (many-to-many relationships)
        if len([col for col in columns if "id" in col.lower()]) >= 2:
            potential["bridge_table_potential"] = True
        
        return potential
    
    def _identify_query_patterns(
        self, 
        table_name: str, 
        data_categories: Dict[str, List[str]]
    ) -> List[str]:
        """Identify query patterns supported by table structure"""
        patterns = []
        
        # Aggregation patterns
        if data_categories["metrics"] and data_categories["categorical"]:
            patterns.append("aggregation_by_category")
        
        # Temporal analysis patterns
        if data_categories["temporal"] and data_categories["metrics"]:
            patterns.append("time_series_analysis")
        
        # Geographical analysis patterns
        if data_categories["geographical"] and data_categories["metrics"]:
            patterns.append("geographical_analysis")
        
        # Hierarchical analysis patterns
        if any(word in table_name.lower() for word in ["territory", "region"]):
            patterns.append("hierarchical_analysis")
        
        # Comparison patterns
        if len(data_categories["metrics"]) > 1:
            patterns.append("metric_comparison")
        
        # Classification patterns
        if data_categories["categorical"] and data_categories["identifiers"]:
            patterns.append("classification_analysis")
        
        return patterns
    
    def _assess_table_complexity(
        self, 
        table_name: str, 
        columns: List[str], 
        data_categories: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Assess complexity indicators of table"""
        complexity = {
            "column_count": len(columns),
            "category_diversity": len([cat for cat, cols in data_categories.items() if cols]),
            "relationship_complexity": "low",
            "query_complexity_potential": "medium",
            "optimization_hints": []
        }
        
        # Assess relationship complexity
        if len(data_categories["relationship"]) > 3:
            complexity["relationship_complexity"] = "high"
            complexity["optimization_hints"].append("consider_join_optimization")
        elif len(data_categories["relationship"]) > 1:
            complexity["relationship_complexity"] = "medium"
        
        # Assess query complexity potential
        if (data_categories["temporal"] and data_categories["metrics"] and 
            data_categories["categorical"]):
            complexity["query_complexity_potential"] = "high"
            complexity["optimization_hints"].append("supports_complex_analytics")
        
        # Column count considerations
        if len(columns) > 50:
            complexity["optimization_hints"].append("consider_column_selection")
        
        return complexity
    
    def _analyze_cross_table_relationships(
        self, 
        table_analyses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze relationships between tables"""
        relationships = {
            "potential_joins": [],
            "shared_dimensions": [],
            "complementary_tables": [],
            "data_flow_patterns": []
        }
        
        table_names = list(table_analyses.keys())
        
        for i, table1 in enumerate(table_names):
            for table2 in table_names[i+1:]:
                # Find potential joins
                joins = self._find_potential_joins(
                    table1, table_analyses[table1],
                    table2, table_analyses[table2]
                )
                relationships["potential_joins"].extend(joins)
                
                # Find shared dimensions
                shared = self._find_shared_dimensions(
                    table1, table_analyses[table1],
                    table2, table_analyses[table2]
                )
                relationships["shared_dimensions"].extend(shared)
        
        return relationships
    
    def _find_potential_joins(
        self, 
        table1: str, 
        analysis1: Dict[str, Any],
        table2: str, 
        analysis2: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find potential join conditions between two tables"""
        joins = []
        
        # Get all columns from both tables
        cols1 = set(col.lower() for col in analysis1["data_categories"]["identifiers"] + 
                   analysis1["data_categories"]["relationship"])
        cols2 = set(col.lower() for col in analysis2["data_categories"]["identifiers"] + 
                   analysis2["data_categories"]["relationship"])
        
        # Find exact matches
        exact_matches = cols1.intersection(cols2)
        for match in exact_matches:
            joins.append({
                "table1": table1,
                "table2": table2,
                "join_column": match,
                "join_type": "exact_match",
                "confidence": 0.9
            })
        
        # Find semantic matches (e.g., territory_id in table1, territoryid in table2)
        semantic_matches = self._find_semantic_column_matches(cols1, cols2)
        for match in semantic_matches:
            joins.append({
                "table1": table1,
                "table2": table2,
                "join_columns": match,
                "join_type": "semantic_match",
                "confidence": 0.7
            })
        
        return joins
    
    def _find_semantic_column_matches(
        self, 
        cols1: Set[str], 
        cols2: Set[str]
    ) -> List[Tuple[str, str]]:
        """Find semantically similar column names between two sets"""
        matches = []
        
        for col1 in cols1:
            for col2 in cols2:
                # Remove common separators and compare
                clean1 = re.sub(r'[_\-\s]', '', col1.lower())
                clean2 = re.sub(r'[_\-\s]', '', col2.lower())
                
                if clean1 == clean2 and col1 != col2:
                    matches.append((col1, col2))
        
        return matches
    
    def _find_shared_dimensions(
        self, 
        table1: str, 
        analysis1: Dict[str, Any],
        table2: str, 
        analysis2: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find shared dimensional attributes between tables"""
        shared = []
        
        # Check for shared categorical dimensions
        cats1 = set(col.lower() for col in analysis1["data_categories"]["categorical"])
        cats2 = set(col.lower() for col in analysis2["data_categories"]["categorical"])
        
        common_cats = cats1.intersection(cats2)
        for cat in common_cats:
            shared.append({
                "table1": table1,
                "table2": table2,
                "shared_dimension": cat,
                "dimension_type": "categorical"
            })
        
        return shared
    
    def _identify_business_domains(self, table_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Identify business domains represented in the schema"""
        domains = {
            "primary_domain": "",
            "secondary_domains": [],
            "domain_coverage": {},
            "cross_domain_opportunities": []
        }
        
        domain_scores = defaultdict(int)
        
        # Score each domain based on table purposes
        for table_name, analysis in table_analyses.items():
            purpose = analysis["business_purpose"].lower()
            
            if any(word in purpose for word in ["healthcare", "pharmaceutical", "prescriber"]):
                domain_scores["healthcare_pharma"] += 1
            elif any(word in purpose for word in ["sales", "marketing", "territory"]):
                domain_scores["sales_marketing"] += 1
            elif any(word in purpose for word in ["financial", "payment", "billing"]):
                domain_scores["financial"] += 1
            elif any(word in purpose for word in ["reporting", "intelligence", "analytics"]):
                domain_scores["business_intelligence"] += 1
        
        # Determine primary and secondary domains
        if domain_scores:
            sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
            domains["primary_domain"] = sorted_domains[0][0]
            domains["secondary_domains"] = [domain for domain, score in sorted_domains[1:] if score > 0]
        
        domains["domain_coverage"] = dict(domain_scores)
        
        return domains
    
    def _determine_query_capabilities(
        self, 
        table_analyses: Dict[str, Any], 
        relationships: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine overall query capabilities of the schema"""
        capabilities = {
            "supported_patterns": set(),
            "analytical_depth": "medium",
            "integration_potential": "medium",
            "recommended_queries": []
        }
        
        # Aggregate supported patterns from all tables
        for table_name, analysis in table_analyses.items():
            capabilities["supported_patterns"].update(analysis["query_patterns"])
        
        # Determine analytical depth
        total_metrics = sum(len(analysis["data_categories"]["metrics"]) 
                          for analysis in table_analyses.values())
        total_temporal = sum(len(analysis["data_categories"]["temporal"]) 
                           for analysis in table_analyses.values())
        
        if total_metrics > 10 and total_temporal > 2:
            capabilities["analytical_depth"] = "high"
        elif total_metrics < 3:
            capabilities["analytical_depth"] = "low"
        
        # Determine integration potential
        join_count = len(relationships["potential_joins"])
        if join_count > 3:
            capabilities["integration_potential"] = "high"
        elif join_count == 0:
            capabilities["integration_potential"] = "low"
        
        # Generate recommended query types
        capabilities["recommended_queries"] = self._generate_recommended_queries(
            capabilities["supported_patterns"], table_analyses
        )
        
        capabilities["supported_patterns"] = list(capabilities["supported_patterns"])
        
        return capabilities
    
    def _generate_recommended_queries(
        self, 
        supported_patterns: Set[str], 
        table_analyses: Dict[str, Any]
    ) -> List[str]:
        """Generate recommended query types based on schema capabilities"""
        recommendations = []
        
        if "aggregation_by_category" in supported_patterns:
            recommendations.append("Count/sum metrics grouped by categorical dimensions")
        
        if "time_series_analysis" in supported_patterns:
            recommendations.append("Trend analysis over time periods")
        
        if "geographical_analysis" in supported_patterns:
            recommendations.append("Geographic distribution and regional analysis")
        
        if "hierarchical_analysis" in supported_patterns:
            recommendations.append("Drill-down analysis by organizational hierarchy")
        
        if "metric_comparison" in supported_patterns:
            recommendations.append("Comparative analysis between different metrics")
        
        if "classification_analysis" in supported_patterns:
            recommendations.append("Classification and segmentation analysis")
        
        return recommendations
    
    def _create_semantic_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create high-level semantic summary of entire schema"""
        summary = {
            "schema_overview": "",
            "key_insights": [],
            "query_recommendations": [],
            "complexity_assessment": "",
            "integration_opportunities": []
        }
        
        table_count = len(analysis["tables"])
        primary_domain = analysis["business_domains"]["primary_domain"]
        
        # Create overview
        summary["schema_overview"] = (
            f"Schema contains {table_count} tables primarily focused on "
            f"{primary_domain.replace('_', ' ')} with "
            f"{len(analysis['cross_table_relationships']['potential_joins'])} potential joins"
        )
        
        # Key insights
        if analysis["query_capabilities"]["analytical_depth"] == "high":
            summary["key_insights"].append("Schema supports complex analytical queries")
        
        if analysis["query_capabilities"]["integration_potential"] == "high":
            summary["key_insights"].append("Tables are well-connected for cross-table analysis")
        
        # Query recommendations
        summary["query_recommendations"] = analysis["query_capabilities"]["recommended_queries"]
        
        # Complexity assessment
        avg_columns = sum(len(table["data_categories"]["identifiers"]) + 
                         len(table["data_categories"]["metrics"]) + 
                         len(table["data_categories"]["categorical"])
                         for table in analysis["tables"].values()) / table_count
        
        if avg_columns > 20:
            summary["complexity_assessment"] = "High complexity - rich schema with many attributes"
        elif avg_columns > 10:
            summary["complexity_assessment"] = "Medium complexity - balanced schema design"
        else:
            summary["complexity_assessment"] = "Low complexity - simple schema structure"
        
        return summary