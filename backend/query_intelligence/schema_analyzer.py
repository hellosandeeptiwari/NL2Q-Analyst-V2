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
                "qty", "units", "amount", "value", "score", "trx", "nrx",
                "tqty", "nqty", "calls", "samples"
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
        
        # Categorize columns by semantic type (extract column names from dict objects)
        column_names = []
        for column in columns:
            if isinstance(column, dict):
                column_name = column.get('name', column.get('column_name', str(column)))
            else:
                column_name = str(column)
            column_names.append(column_name)
        
        table_analysis["data_categories"] = self._categorize_columns_semantically(column_names)
        
        # Analyze individual column semantics
        for column in columns:
            # Extract column name to use as dictionary key (must be hashable)
            if isinstance(column, dict):
                column_key = column.get('name', column.get('column_name', str(column)))
            else:
                column_key = str(column)
            
            table_analysis["column_semantics"][column_key] = self._analyze_column_semantics(
                column_key, table_name
            )
        
        # Assess relationship potential
        table_analysis["relationship_potential"] = self._assess_relationship_potential(
            table_name, column_names
        )
        
        # Identify supported query patterns
        table_analysis["query_patterns"] = self._identify_query_patterns(
            table_name, table_analysis["data_categories"]
        )
        
        # Assess complexity indicators
        table_analysis["complexity_indicators"] = self._assess_table_complexity(
            table_name, columns, table_analysis["data_categories"]
        )
        
        # 🔧 CRITICAL FIX: Preserve the actual columns in the analysis result
        # This was missing and causing the LLM to receive empty column arrays!
        table_analysis["columns"] = columns
        
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
            # Handle both string and dict column formats
            if isinstance(column, dict):
                column_name = column.get('name', '')
            else:
                column_name = str(column)
            column_lower = column_name.lower()
            categorized = False
            
            # Check each pattern type
            for category, patterns in self.column_type_patterns.items():
                if any(pattern in column_lower for pattern in patterns):
                    # Map pattern keys to category keys (handle singular/plural mismatches)
                    if category == 'identifier':
                        categories['identifiers'].append(column_name)
                    elif category == 'metric':
                        categories['metrics'].append(column_name)  # Fix: metric -> metrics
                    else:
                        categories[category].append(column_name)
                    categorized = True
                    break
            
            # Special case for relationship columns (foreign keys)
            if any(word in column_lower for word in ["id", "key"]) and "_" in column_lower:
                categories["relationship"].append(column_name)
                categorized = True
            
            # If not categorized, add to descriptive
            if not categorized:
                categories["descriptive"].append(column_name)
        
        return categories
    
    def _analyze_column_semantics(self, column: str, table_name: str) -> Dict[str, Any]:
        """Analyze semantic meaning of individual column"""
        # Handle both string and dict column formats
        if isinstance(column, dict):
            column_name = column.get('name', column.get('column_name', ''))
        else:
            column_name = str(column)
            
        column_lower = column_name.lower()
        table_lower = table_name.lower()
        
        semantics = {
            "column_name": column_name,
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
        id_columns = []
        for col in columns:
            # Handle both string and dict column formats
            if isinstance(col, dict):
                col_name = col.get('name', col.get('column_name', ''))
            else:
                col_name = str(col)
            if "id" in col_name.lower():
                id_columns.append(col_name)
        
        if len(id_columns) >= 2:
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
        def extract_column_name(col):
            if isinstance(col, dict):
                return col.get('name', str(col))
            return str(col)
        
        cols1 = set(extract_column_name(col).lower() for col in analysis1["data_categories"]["identifiers"] + 
                   analysis1["data_categories"]["relationship"])
        cols2 = set(extract_column_name(col).lower() for col in analysis2["data_categories"]["identifiers"] + 
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
        def extract_column_name(col):
            if isinstance(col, dict):
                return col.get('name', str(col))
            return str(col)
            
        cats1 = set(extract_column_name(col).lower() for col in analysis1["data_categories"]["categorical"])
        cats2 = set(extract_column_name(col).lower() for col in analysis2["data_categories"]["categorical"])
        
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
    
    def analyze_table_semantics(self, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous wrapper for table semantic analysis.
        Used by IntelligentQueryPlanner for compatibility.
        
        Args:
            table_info: Dictionary containing table metadata
            
        Returns:
            Table semantic analysis results
        """
        try:
            table_name = table_info.get('table_name', 'unknown')
            columns = table_info.get('columns', [])
            # If columns is a list, convert to the expected format
            if isinstance(columns, list):
                columns_dict = {}
                for col in columns:
                    # Extract column name to use as dictionary key (must be hashable)
                    if isinstance(col, dict):
                        col_key = col.get('name', col.get('column_name', str(col)))
                    else:
                        col_key = str(col)
                    columns_dict[col_key] = 'varchar'
            else:
                columns_dict = columns
            
            # Create basic semantic analysis (synchronous version)
            analysis = {
                "table_name": table_name,
                "business_purpose": self._infer_table_business_purpose(table_name), 
                "data_categories": self._categorize_columns_semantically(list(columns_dict.keys())),
                "domain_entities": self._extract_domain_entities(table_name, list(columns_dict.keys())),
                "relationship_types": self._identify_relationship_types(table_name),
                "query_patterns": self._identify_query_patterns(table_name, self._categorize_columns_semantically(list(columns_dict.keys()))),
                "complexity_score": len(columns_dict) * 0.1  # Simple complexity metric
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in synchronous table semantic analysis: {e}")
            # Return minimal fallback analysis
            return {
                "table_name": table_info.get('table_name', 'unknown'),
                "business_purpose": "Data storage",
                "data_categories": {},
                "domain_entities": [],
                "relationship_types": [],
                "query_patterns": [],
                "complexity_score": 0.5
            }
    
    def _extract_domain_entities(self, table_name: str, columns: List[str]) -> List[str]:
        """Extract domain entities from table name and columns"""
        entities = []
        
        # Extract from table name
        table_words = table_name.lower().replace('_', ' ').split()
        for word in table_words:
            if word in ['user', 'customer', 'product', 'order', 'sales', 'employee', 'account']:
                entities.append(word)
        
        # Extract from column names
        for col in columns:
            # Handle both string and dict column formats
            if isinstance(col, dict):
                col_name = col.get('name', col.get('column_name', ''))
            else:
                col_name = str(col)
            
            col_lower = col_name.lower()
            if any(entity in col_lower for entity in ['customer', 'product', 'order', 'user', 'account']):
                entities.extend([entity for entity in ['customer', 'product', 'order', 'user', 'account'] if entity in col_lower])
        
        return list(set(entities))
    
    def _identify_relationship_types(self, table_name: str) -> List[str]:
        """Identify potential relationship types for the table"""
        relationships = []
        
        table_lower = table_name.lower()
        if 'profile' in table_lower or 'prescriber' in table_lower:
            relationships.extend(['one-to-many', 'joins'])
        if 'sales' in table_lower or 'order' in table_lower:
            relationships.extend(['transactional', 'temporal'])
        if 'product' in table_lower:
            relationships.extend(['categorical', 'hierarchical'])
            
        return relationships if relationships else ['general']
    
    def _classify_column_semantic_type(self, column_name: str) -> str:
        """
        Classify a column into semantic types for intelligent query planning.
        
        Args:
            column_name: The name of the column to classify
            
        Returns:
            Semantic type classification
        """
        column_lower = column_name.lower()
        
        # Measure columns (metrics, values, counts)
        if any(pattern in column_lower for pattern in ['trx', 'nrx', 'qty', 'count', 'amount', 'revenue', 'sales', 'calls', 'samples']):
            return 'measure'
        
        # Key columns (IDs, identifiers)
        if any(pattern in column_lower for pattern in ['id', '_key', 'number', 'code']) and not any(pattern in column_lower for pattern in ['name', 'description']):
            return 'key'
        
        # Date/Time columns
        if any(pattern in column_lower for pattern in ['date', 'time', 'period', 'month', 'year', 'quarter']):
            return 'temporal'
        
        # Geographic columns
        if any(pattern in column_lower for pattern in ['territory', 'region', 'state', 'city', 'zip', 'address', 'location']):
            return 'geographic'
        
        # Product/Entity columns
        if any(pattern in column_lower for pattern in ['product', 'drug', 'medication', 'brand']):
            return 'product'
        
        # Person/Entity columns
        if any(pattern in column_lower for pattern in ['prescriber', 'patient', 'doctor', 'physician', 'name', 'provider']):
            return 'entity'
        
        # Boolean/Flag columns
        if any(pattern in column_lower for pattern in ['flag', 'target', 'active', 'enabled', 'include']):
            return 'boolean'
        
        # Classification/Category columns
        if any(pattern in column_lower for pattern in ['type', 'category', 'class', 'group', 'tier', 'priority', 'specialty']):
            return 'categorical'
        
        # Default to attribute
        return 'attribute'
    
    def find_potential_joins(self, tables_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Public method to find potential joins between multiple tables using table_info structures.
        This is called by the IntelligentQueryPlanner and handles multi-table scenarios.
        
        Args:
            tables_info: List of dictionaries containing table metadata including columns
            
        Returns:
            List of potential join relationships between all the tables
        """
        try:
            if not tables_info or len(tables_info) < 2:
                print("🔗 Need at least 2 tables to find joins")
                return []
            
            all_joins = []
            table_names = [table.get('table_name', f'table_{i}') for i, table in enumerate(tables_info)]
            
            print(f"🔗 Finding potential joins between {len(tables_info)} tables: {', '.join(table_names)}")
            
            # Create analysis structures for all tables
            table_analyses = {}
            for table_info in tables_info:
                table_name = table_info.get('table_name', 'unknown')
                columns = table_info.get('columns', [])
                table_analyses[table_name] = self._create_join_analysis_structure(columns, table_name)
            
            # Find joins between every pair of tables
            for i in range(len(tables_info)):
                for j in range(i + 1, len(tables_info)):
                    table1_name = table_names[i]
                    table2_name = table_names[j]
                    
                    analysis1 = table_analyses[table1_name]
                    analysis2 = table_analyses[table2_name]
                    
                    # Use the existing private method to find joins between this pair
                    pair_joins = self._find_potential_joins(table1_name, analysis1, table2_name, analysis2)
                    all_joins.extend(pair_joins)
            
            # Also look for multi-table joins (3+ tables connected through common keys)
            multi_joins = self._find_multi_table_joins(table_analyses)
            all_joins.extend(multi_joins)
            
            print(f"🔗 Found {len(all_joins)} total potential joins across all tables")
            
            return all_joins
            
        except Exception as e:
            print(f"❌ Error finding potential joins: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_join_analysis_structure(self, columns: List[Dict[str, Any]], table_name: str) -> Dict[str, Any]:
        """
        Create analysis structure from columns for join detection.
        Categorizes columns into identifiers and relationships based on naming patterns.
        """
        identifiers = []
        relationships = []
        
        for col in columns:
            column_name = ""
            
            # Handle different column formats
            if isinstance(col, dict):
                column_name = col.get('column_name', '')
            elif isinstance(col, str):
                column_name = col
            else:
                continue
                
            if not column_name:
                continue
                
            column_lower = column_name.lower()
            
            # Classify as identifier (primary/foreign keys)
            if any(keyword in column_lower for keyword in ['id', 'key', 'code', 'number', 'nbr']):
                identifiers.append(column_name)
            # Classify as relationship column
            elif any(keyword in column_lower for keyword in ['territory', 'region', 'area', 'group', 'type', 'category']):
                relationships.append(column_name)
            # Add common pharmaceutical domain patterns
            elif any(keyword in column_lower for keyword in ['prescriber', 'provider', 'product', 'specialty', 'npi']):
                relationships.append(column_name)
        
        print(f"🔗 {table_name}: {len(identifiers)} identifiers, {len(relationships)} relationships")
        
        return {
            "data_categories": {
                "identifiers": identifiers,
                "relationship": relationships
            }
        }
    
    def _find_multi_table_joins(self, table_analyses: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find multi-table joins where 3+ tables can be connected through common keys.
        This identifies chain joins like Table1 -> Table2 -> Table3.
        
        Args:
            table_analyses: Dictionary of table_name -> analysis structure
            
        Returns:
            List of multi-table join relationships
        """
        multi_joins = []
        
        try:
            table_names = list(table_analyses.keys())
            
            if len(table_names) < 3:
                return multi_joins
            
            # Look for common identifiers across multiple tables
            identifier_map = {}  # identifier -> list of tables that have it
            
            for table_name, analysis in table_analyses.items():
                identifiers = analysis.get("data_categories", {}).get("identifiers", [])
                for identifier in identifiers:
                    identifier_lower = identifier.lower()
                    # Normalize common patterns
                    normalized_id = self._normalize_identifier(identifier_lower)
                    
                    if normalized_id not in identifier_map:
                        identifier_map[normalized_id] = []
                    identifier_map[normalized_id].append({
                        'table': table_name,
                        'column': identifier
                    })
            
            # Find identifiers that appear in 3+ tables (potential chain joins)
            for normalized_id, table_columns in identifier_map.items():
                if len(table_columns) >= 3:
                    # Create multi-table join relationship
                    multi_join = {
                        "join_type": "multi_table_chain",
                        "common_key": normalized_id,
                        "tables": table_columns,
                        "confidence": 0.8,  # High confidence for multi-table keys
                        "description": f"Chain join through common key '{normalized_id}' across {len(table_columns)} tables"
                    }
                    multi_joins.append(multi_join)
                    
                    print(f"🔗 Multi-table join found: {normalized_id} connects {len(table_columns)} tables")
            
        except Exception as e:
            print(f"❌ Error in multi-table join analysis: {e}")
        
        return multi_joins
    
    def _normalize_identifier(self, identifier: str) -> str:
        """
        Normalize identifier names to detect common patterns across tables.
        
        Args:
            identifier: Column name to normalize
            
        Returns:
            Normalized identifier string
        """
        # Remove common prefixes/suffixes and patterns
        patterns_to_remove = ['_id', '_key', '_code', '_nbr', '_number']
        
        normalized = identifier.lower().strip()
        
        # Remove common patterns
        for pattern in patterns_to_remove:
            if normalized.endswith(pattern):
                normalized = normalized[:-len(pattern)]
        
        # Handle common pharmaceutical domain identifiers
        if 'prescriber' in normalized:
            return 'prescriber'
        elif 'provider' in normalized:
            return 'provider'
        elif 'territory' in normalized:
            return 'territory'
        elif 'product' in normalized:
            return 'product'
        elif 'npi' in normalized:
            return 'npi'
        
        return normalized