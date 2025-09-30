"""
Intelligent Query Planner
========================

This module handles intelligent query planning by:
1. Understanding all available schema semantics
2. Creating comprehensive query plans without artificial limitations
3. Letting LLM decide table usage based on actual relevance
4. Always preserving SQL for UI display (success or failure)

Architecture:
- Schema Semantic Understanding: Gets ALL indexed tables with full metadata
- Intelligent Query Planning: LLM analyzes user query against complete schema
- Query Generation Planning: Creates plan with semantic understanding
- UI Transparency: Always shows generated SQL regardless of execution result
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class IntelligentQueryPlanner:
    """
    Intelligent query planner that understands schema semantics and creates
    comprehensive query plans without artificial simple/complex limitations.
    """
    
    def __init__(self):
        self.logger = logger
        
    async def get_all_indexed_schema_semantics(self, pinecone_store) -> Dict[str, Any]:
        """
        Get complete schema semantics for ALL indexed tables.
        No filtering - let LLM understand full context.
        
        Returns:
            Dict containing all table metadata with semantic understanding
        """
        try:
            self.logger.info("🧠 Getting complete schema semantics for ALL indexed tables")
            
            # Get all indexed tables from Pinecone
            indexed_tables = await pinecone_store._get_indexed_tables_fast()
            self.logger.info(f"📊 Found {len(indexed_tables)} indexed tables: {list(indexed_tables)}")
            
            schema_semantics = {
                "indexed_tables": list(indexed_tables),
                "table_metadata": {},
                "semantic_relationships": {},
                "business_context": {}
            }
            
            # Get complete metadata for each indexed table
            for table_name in indexed_tables:
                try:
                    # Get complete table details from Pinecone
                    table_columns = await pinecone_store._get_table_columns_from_pinecone(table_name)
                    
                    if table_columns:
                        schema_semantics["table_metadata"][table_name] = {
                            "columns": table_columns,
                            "column_count": len(table_columns),
                            "semantic_hints": self._extract_semantic_hints(table_name, table_columns),
                            "business_domain": self._infer_business_domain(table_name, table_columns),
                            "data_patterns": self._identify_data_patterns(table_columns)
                        }
                        self.logger.info(f"✅ Retrieved {len(table_columns)} columns for {table_name}")
                    else:
                        self.logger.warning(f"⚠️ No columns found for {table_name}")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error getting metadata for {table_name}: {e}")
                    continue
            
            # Identify potential relationships between tables
            schema_semantics["semantic_relationships"] = self._identify_table_relationships(
                schema_semantics["table_metadata"]
            )
            
            self.logger.info(f"🎯 Complete schema semantics prepared for {len(schema_semantics['table_metadata'])} tables")
            return schema_semantics
            
        except Exception as e:
            self.logger.error(f"❌ Error getting schema semantics: {e}")
            return {"indexed_tables": [], "table_metadata": {}, "semantic_relationships": {}, "business_context": {}}
    
    def _extract_semantic_hints(self, table_name: str, columns: List[str]) -> Dict[str, Any]:
        """Extract semantic hints from table and column names"""
        hints = {
            "temporal_columns": [],
            "identifier_columns": [],
            "metric_columns": [],
            "categorical_columns": [],
            "relationship_columns": []
        }
        
        for col in columns:
            col_lower = col.lower()
            
            # Temporal indicators
            if any(word in col_lower for word in ['date', 'time', 'period', 'start', 'end']):
                hints["temporal_columns"].append(col)
            
            # Identifier indicators  
            elif any(word in col_lower for word in ['id', 'key', 'number', 'code']):
                hints["identifier_columns"].append(col)
            
            # Metric indicators
            elif any(word in col_lower for word in ['count', 'total', 'sum', 'avg', 'rate', 'trx', 'nrx', 'qty']):
                hints["metric_columns"].append(col)
            
            # Categorical indicators
            elif any(word in col_lower for word in ['name', 'type', 'category', 'flag', 'tier', 'status']):
                hints["categorical_columns"].append(col)
        
        return hints
    
    def _infer_business_domain(self, table_name: str, columns: List[str]) -> str:
        """Infer business domain from table and column patterns"""
        table_lower = table_name.lower()
        all_columns_lower = ' '.join(columns).lower()
        
        # Healthcare/Pharma patterns
        if any(word in table_lower + all_columns_lower for word in 
               ['prescriber', 'patient', 'drug', 'pharma', 'medical', 'nrx', 'trx']):
            return "Healthcare/Pharmaceutical"
        
        # Financial patterns
        elif any(word in table_lower + all_columns_lower for word in 
                ['payment', 'invoice', 'billing', 'revenue', 'cost']):
            return "Financial"
        
        # Sales/Marketing patterns
        elif any(word in table_lower + all_columns_lower for word in 
                ['territory', 'region', 'sales', 'marketing', 'target']):
            return "Sales/Marketing"
        
        # Reporting/Analytics patterns
        elif any(word in table_lower for word in ['reporting', 'bi', 'analytics']):
            return "Business Intelligence/Reporting"
        
        return "General Business"
    
    def _identify_data_patterns(self, columns: List[str]) -> Dict[str, Any]:
        """Identify common data patterns in columns"""
        patterns = {
            "has_temporal_data": False,
            "has_hierarchical_data": False,
            "has_metrics": False,
            "has_geographical_data": False,
            "complexity_indicators": []
        }
        
        all_columns_lower = ' '.join(columns).lower()
        
        # Temporal patterns
        if any(word in all_columns_lower for word in ['date', 'period', 'week', 'month', 'quarter']):
            patterns["has_temporal_data"] = True
            patterns["complexity_indicators"].append("temporal_analysis")
        
        # Hierarchical patterns
        if any(word in all_columns_lower for word in ['region', 'territory', 'area', 'zone']):
            patterns["has_hierarchical_data"] = True
            patterns["complexity_indicators"].append("hierarchical_data")
        
        # Metric patterns
        if any(word in all_columns_lower for word in ['trx', 'nrx', 'qty', 'count', 'total', 'share']):
            patterns["has_metrics"] = True
            patterns["complexity_indicators"].append("metric_analysis")
        
        # Geographical patterns
        if any(word in all_columns_lower for word in ['city', 'state', 'zip', 'address', 'location']):
            patterns["has_geographical_data"] = True
            patterns["complexity_indicators"].append("geographical_analysis")
        
        return patterns
    
    def _identify_table_relationships(self, table_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Identify potential relationships between tables based on column patterns"""
        relationships = {
            "potential_joins": [],
            "common_dimensions": [],
            "complementary_tables": []
        }
        
        table_names = list(table_metadata.keys())
        
        for i, table1 in enumerate(table_names):
            for table2 in table_names[i+1:]:
                # Find common column patterns
                cols1 = set(col.lower() for col in table_metadata[table1]["columns"])
                cols2 = set(col.lower() for col in table_metadata[table2]["columns"])
                
                common_cols = cols1.intersection(cols2)
                
                if common_cols:
                    relationships["potential_joins"].append({
                        "table1": table1,
                        "table2": table2,
                        "common_columns": list(common_cols),
                        "join_strength": len(common_cols)
                    })
                
                # Check for complementary business domains
                domain1 = table_metadata[table1]["business_domain"]
                domain2 = table_metadata[table2]["business_domain"]
                
                if domain1 == domain2 and domain1 != "General Business":
                    relationships["complementary_tables"].append({
                        "table1": table1,
                        "table2": table2,
                        "domain": domain1,
                        "synergy": "same_domain"
                    })
        
        return relationships
    
    async def create_intelligent_query_plan(
        self, 
        user_query: str, 
        schema_semantics: Dict[str, Any],
        conversation_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Create intelligent query plan based on complete schema understanding.
        NO artificial simple/complex limitations.
        
        Args:
            user_query: User's natural language query
            schema_semantics: Complete schema semantic understanding
            conversation_context: Previous conversation context if any
            
        Returns:
            Comprehensive query plan with semantic understanding
        """
        try:
            self.logger.info(f"🧠 Creating intelligent query plan for: '{user_query}'")
            
            query_plan = {
                "user_query": user_query,
                "timestamp": datetime.now().isoformat(),
                "semantic_analysis": {},
                "table_selection": {},
                "query_strategy": {},
                "execution_approach": {},
                "ui_requirements": {
                    "always_show_sql": True,  # CRITICAL: Always show SQL regardless of success/failure
                    "show_plan_reasoning": True,
                    "enable_debugging": True
                }
            }
            
            # Analyze query semantics
            query_plan["semantic_analysis"] = self._analyze_query_semantics(
                user_query, schema_semantics
            )
            
            # Select relevant tables based on semantic understanding
            query_plan["table_selection"] = self._select_tables_intelligently(
                user_query, schema_semantics, query_plan["semantic_analysis"]
            )
            
            # Determine query strategy
            query_plan["query_strategy"] = self._determine_query_strategy(
                user_query, query_plan["table_selection"], schema_semantics
            )
            
            # Plan execution approach
            query_plan["execution_approach"] = self._plan_execution_approach(
                query_plan["query_strategy"], conversation_context
            )
            
            self.logger.info(f"✅ Intelligent query plan created with {len(query_plan['table_selection']['selected_tables'])} tables")
            return query_plan
            
        except Exception as e:
            self.logger.error(f"❌ Error creating query plan: {e}")
            return {
                "user_query": user_query,
                "error": str(e),
                "ui_requirements": {"always_show_sql": True}
            }
    
    def _analyze_query_semantics(self, user_query: str, schema_semantics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze query semantics against available schema"""
        analysis = {
            "query_intent": [],
            "data_requirements": [],
            "complexity_factors": [],
            "visualization_needed": False
        }
        
        query_lower = user_query.lower()
        
        # Identify query intent
        if any(word in query_lower for word in ['count', 'how many', 'number of']):
            analysis["query_intent"].append("aggregation")
        
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']):
            analysis["query_intent"].append("comparison")
        
        if any(word in query_lower for word in ['trend', 'over time', 'growth', 'decline']):
            analysis["query_intent"].append("temporal_analysis")
        
        if any(word in query_lower for word in ['new', 'grower', 'decliner', 'category']):
            analysis["query_intent"].append("classification")
        
        # Check for visualization request
        if any(word in query_lower for word in ['chart', 'graph', 'plot', 'visualize', 'show']):
            analysis["visualization_needed"] = True
        
        # Identify data requirements
        for table_name, metadata in schema_semantics["table_metadata"].items():
            semantic_hints = metadata["semantic_hints"]
            
            if "temporal_analysis" in analysis["query_intent"] and semantic_hints["temporal_columns"]:
                analysis["data_requirements"].append(f"{table_name}_temporal")
            
            if "aggregation" in analysis["query_intent"] and semantic_hints["metric_columns"]:
                analysis["data_requirements"].append(f"{table_name}_metrics")
            
            if "classification" in analysis["query_intent"] and semantic_hints["categorical_columns"]:
                analysis["data_requirements"].append(f"{table_name}_categories")
        
        return analysis
    
    def _select_tables_intelligently(
        self, 
        user_query: str, 
        schema_semantics: Dict[str, Any], 
        semantic_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Select tables based on intelligent semantic matching - NO artificial limitations"""
        
        selection = {
            "selected_tables": [],
            "selection_reasoning": [],
            "potential_joins": [],
            "confidence_scores": {}
        }
        
        # Score each table based on semantic relevance
        for table_name, metadata in schema_semantics["table_metadata"].items():
            score = self._calculate_table_relevance_score(
                user_query, table_name, metadata, semantic_analysis
            )
            
            selection["confidence_scores"][table_name] = score
            
            # Include table if it has meaningful relevance (threshold: 0.3)
            if score > 0.3:
                selection["selected_tables"].append(table_name)
                selection["selection_reasoning"].append(
                    f"{table_name}: score {score:.2f} - {self._explain_table_selection(table_name, metadata, semantic_analysis)}"
                )
        
        # Sort tables by relevance score
        selection["selected_tables"].sort(
            key=lambda table: selection["confidence_scores"][table], 
            reverse=True
        )
        
        # Identify potential joins between selected tables
        for relationship in schema_semantics["semantic_relationships"]["potential_joins"]:
            if (relationship["table1"] in selection["selected_tables"] and 
                relationship["table2"] in selection["selected_tables"]):
                selection["potential_joins"].append(relationship)
        
        self.logger.info(f"🎯 Selected {len(selection['selected_tables'])} tables based on semantic relevance")
        return selection
    
    def _calculate_table_relevance_score(
        self, 
        user_query: str, 
        table_name: str, 
        metadata: Dict[str, Any], 
        semantic_analysis: Dict[str, Any]
    ) -> float:
        """Calculate relevance score for a table based on semantic matching"""
        score = 0.0
        query_lower = user_query.lower()
        table_lower = table_name.lower()
        
        # Table name relevance
        query_words = set(query_lower.split())
        table_words = set(table_lower.replace('_', ' ').split())
        
        word_overlap = len(query_words.intersection(table_words))
        if word_overlap > 0:
            score += 0.4 * (word_overlap / len(query_words))
        
        # Column semantic relevance
        semantic_hints = metadata["semantic_hints"]
        
        if "temporal_analysis" in semantic_analysis["query_intent"] and semantic_hints["temporal_columns"]:
            score += 0.3
        
        if "aggregation" in semantic_analysis["query_intent"] and semantic_hints["metric_columns"]:
            score += 0.3
        
        if "classification" in semantic_analysis["query_intent"] and semantic_hints["categorical_columns"]:
            score += 0.2
        
        # Business domain relevance
        domain_keywords = {
            "Healthcare/Pharmaceutical": ["prescriber", "patient", "drug", "medical"],
            "Sales/Marketing": ["territory", "region", "sales", "target"],
            "Business Intelligence/Reporting": ["report", "analysis", "bi", "overview"]
        }
        
        domain = metadata["business_domain"]
        if domain in domain_keywords:
            domain_words = domain_keywords[domain]
            if any(word in query_lower for word in domain_words):
                score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _explain_table_selection(
        self, 
        table_name: str, 
        metadata: Dict[str, Any], 
        semantic_analysis: Dict[str, Any]
    ) -> str:
        """Explain why a table was selected"""
        reasons = []
        
        if metadata["semantic_hints"]["temporal_columns"] and "temporal_analysis" in semantic_analysis["query_intent"]:
            reasons.append("has temporal data")
        
        if metadata["semantic_hints"]["metric_columns"] and "aggregation" in semantic_analysis["query_intent"]:
            reasons.append("has metrics")
        
        if metadata["semantic_hints"]["categorical_columns"] and "classification" in semantic_analysis["query_intent"]:
            reasons.append("has categories")
        
        reasons.append(f"domain: {metadata['business_domain']}")
        
        return ", ".join(reasons)
    
    def _determine_query_strategy(
        self, 
        user_query: str, 
        table_selection: Dict[str, Any], 
        schema_semantics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine comprehensive query strategy"""
        strategy = {
            "approach": "intelligent_semantic",
            "table_usage": table_selection["selected_tables"],
            "join_strategy": [],
            "aggregation_needed": False,
            "grouping_dimensions": [],
            "temporal_analysis": False,
            "complexity_level": "adaptive"  # No more artificial simple/complex
        }
        
        # Determine join strategy if multiple tables
        if len(table_selection["selected_tables"]) > 1:
            strategy["join_strategy"] = table_selection["potential_joins"]
        
        # Determine aggregation needs
        if any(word in user_query.lower() for word in ['count', 'sum', 'total', 'how many']):
            strategy["aggregation_needed"] = True
        
        # Determine grouping dimensions
        for table_name in table_selection["selected_tables"]:
            if table_name in schema_semantics["table_metadata"]:
                categorical_cols = schema_semantics["table_metadata"][table_name]["semantic_hints"]["categorical_columns"]
                strategy["grouping_dimensions"].extend(categorical_cols)
        
        # Check for temporal analysis
        if any(word in user_query.lower() for word in ['trend', 'over time', 'growth', 'decline', 'new', 'grower']):
            strategy["temporal_analysis"] = True
        
        return strategy
    
    def _plan_execution_approach(
        self, 
        query_strategy: Dict[str, Any], 
        conversation_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Plan execution approach with UI transparency"""
        approach = {
            "execution_mode": "transparent",
            "error_handling": "preserve_sql",
            "ui_display": {
                "always_show_sql": True,  # CRITICAL
                "show_execution_plan": True,
                "show_error_details": True,
                "preserve_failed_queries": True
            },
            "fallback_strategy": "retry_with_simpler_joins",
            "debug_mode": True
        }
        
        # Adaptive execution based on strategy
        if len(query_strategy["table_usage"]) > 2:
            approach["join_optimization"] = True
            approach["execution_timeout"] = 30  # seconds
        else:
            approach["execution_timeout"] = 15  # seconds
        
        if query_strategy["temporal_analysis"]:
            approach["temporal_optimization"] = True
        
        return approach