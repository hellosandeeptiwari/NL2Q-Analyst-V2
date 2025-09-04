"""
Schema Discovery Tool - Intelligent schema exploration and mapping
Implements auto-discovery of tables, columns, joins, enums, date grains, metrics
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
from datetime import datetime, timedelta

from backend.db.engine import get_adapter
from backend.agents.openai_vector_matcher import OpenAIVectorMatcher

@dataclass
class TableInfo:
    name: str
    schema: str
    type: str  # 'table', 'view', 'materialized_view'
    columns: List[Dict[str, Any]]
    row_count: Optional[int] = None
    last_updated: Optional[datetime] = None
    description: Optional[str] = None
    is_fact_table: bool = False
    is_dimension_table: bool = False
    primary_keys: List[str] = None
    foreign_keys: List[Dict[str, str]] = None

@dataclass
class ColumnInfo:
    name: str
    data_type: str
    is_nullable: bool
    description: Optional[str] = None
    sample_values: List[Any] = None
    cardinality: Optional[int] = None
    is_pii: bool = False
    is_key: bool = False
    business_synonyms: List[str] = None

@dataclass
class SchemaContext:
    relevant_tables: List[TableInfo]
    entity_mappings: Dict[str, List[str]]  # entity -> column names
    join_paths: List[Dict[str, Any]]
    metrics_available: List[Dict[str, Any]]
    date_columns: List[Dict[str, Any]]
    filter_suggestions: List[Dict[str, Any]]
    business_glossary: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert SchemaContext to a JSON-serializable dictionary"""
        def serialize_table(table: TableInfo) -> Dict[str, Any]:
            return {
                "name": table.name,
                "schema": table.schema,
                "type": table.type,
                "columns": table.columns,
                "row_count": table.row_count,
                "last_updated": table.last_updated.isoformat() if table.last_updated else None,
                "description": table.description,
                "is_fact_table": table.is_fact_table,
                "is_dimension_table": table.is_dimension_table,
                "primary_keys": table.primary_keys or [],
                "foreign_keys": table.foreign_keys or [],
                "business_synonyms": table.business_synonyms or []
            }
        
        return {
            "relevant_tables": [serialize_table(table) for table in self.relevant_tables],
            "entity_mappings": self.entity_mappings,
            "join_paths": self.join_paths,
            "metrics_available": self.metrics_available,
            "date_columns": self.date_columns,
            "filter_suggestions": self.filter_suggestions,
            "business_glossary": self.business_glossary
        }

class SchemaTool:
    """
    Advanced schema discovery tool that understands business context
    """
    
    def __init__(self):
        self.vector_matcher = OpenAIVectorMatcher()
        self.db_adapter = get_adapter()
        
        # Business domain mappings
        self.business_synonyms = {
            "revenue": ["sales", "income", "earnings", "turnover"],
            "customer": ["client", "account", "user", "subscriber"],
            "product": ["item", "sku", "offering", "service"],
            "date": ["time", "period", "when", "timestamp"],
            "location": ["geography", "region", "area", "territory"],
            "count": ["number", "total", "quantity", "volume"]
        }
        
        # Common table patterns
        self.table_patterns = {
            "fact": ["sales", "transactions", "events", "facts", "measures"],
            "dimension": ["dim_", "lookup", "master", "reference"],
            "temporal": ["daily", "monthly", "quarterly", "yearly", "time"]
        }
        
    async def discover_schema(self, query: str, entities: List[str] = None) -> SchemaContext:
        """
        Main entry point for schema discovery
        """
        
        # Step 1: Extract entities and intent from query
        extracted_entities = await self._extract_entities_from_query(query)
        all_entities = list(set((entities or []) + extracted_entities))
        
        # Step 2: Find relevant tables using vector similarity
        relevant_tables = await self._find_relevant_tables(query, all_entities)
        
        # Step 3: Analyze table relationships and join paths
        join_paths = await self._discover_join_paths(relevant_tables)
        
        # Step 4: Map entities to specific columns
        entity_mappings = await self._map_entities_to_columns(all_entities, relevant_tables)
        
        # Step 5: Identify available metrics and aggregations
        metrics_available = await self._discover_metrics(relevant_tables, query)
        
        # Step 6: Find date/time columns for filtering
        date_columns = await self._discover_date_columns(relevant_tables)
        
        # Step 7: Generate filter suggestions
        filter_suggestions = await self._generate_filter_suggestions(query, relevant_tables)
        
        # Step 8: Build business glossary context
        business_glossary = await self._build_business_glossary(all_entities)
        
        return SchemaContext(
            relevant_tables=relevant_tables,
            entity_mappings=entity_mappings,
            join_paths=join_paths,
            metrics_available=metrics_available,
            date_columns=date_columns,
            filter_suggestions=filter_suggestions,
            business_glossary=business_glossary
        )
    
    async def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract business entities mentioned in the query"""
        
        entities = []
        query_lower = query.lower()
        
        # Common business entities patterns
        entity_patterns = {
            "customer": ["customer", "client", "account", "user"],
            "product": ["product", "item", "sku", "service"],
            "sales": ["sales", "revenue", "purchase", "order"],
            "time": ["day", "week", "month", "quarter", "year", "date"],
            "location": ["region", "state", "country", "territory", "geography"],
            "employee": ["employee", "staff", "rep", "agent", "salesperson"]
        }
        
        for entity_type, patterns in entity_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    entities.append(entity_type)
                    break
        
        # Extract potential table/column names mentioned directly
        # Look for capitalized words that might be table names
        capitalized_words = re.findall(r'\b[A-Z][a-zA-Z_]*\b', query)
        entities.extend(capitalized_words)
        
        return list(set(entities))
    
    async def _find_relevant_tables(self, query: str, entities: List[str]) -> List[TableInfo]:
        """Find tables relevant to the query using vector similarity"""
        
        try:
            # Get enhanced schema from cache
            from backend.db.enhanced_schema import get_enhanced_schema_cache
            schema_cache = get_enhanced_schema_cache()
            
            # Use vector matcher to find relevant tables
            relevant_matches = self.vector_matcher.find_similar_tables(
                query, 
                top_k=10
            )
            
            relevant_tables = []
            
            for match in relevant_matches:
                table_name = match.get("table_name", "")
                if table_name in schema_cache:
                    table_info = schema_cache[table_name]
                    
                    # Extract column information
                    columns = []
                    for col_name, col_info in table_info.get("columns", {}).items():
                        column = ColumnInfo(
                            name=col_name,
                            data_type=col_info.get("data_type", "unknown"),
                            is_nullable=col_info.get("nullable", True),
                            description=col_info.get("description"),
                            sample_values=col_info.get("sample_values", []),
                            cardinality=col_info.get("cardinality"),
                            is_pii=self._is_pii_column(col_name),
                            is_key=self._is_key_column(col_name, col_info),
                            business_synonyms=self._get_business_synonyms(col_name)
                        )
                        columns.append(column)
                    
                    table = TableInfo(
                        name=table_name,
                        schema=table_info.get("schema", "public"),
                        type=table_info.get("table_type", "table"),
                        columns=columns,
                        row_count=table_info.get("row_count"),
                        description=table_info.get("description"),
                        is_fact_table=self._is_fact_table(table_name, columns),
                        is_dimension_table=self._is_dimension_table(table_name, columns)
                    )
                    
                    relevant_tables.append(table)
            
            return relevant_tables
            
        except Exception as e:
            print(f"Error finding relevant tables: {e}")
            return []
    
    async def _discover_join_paths(self, tables: List[TableInfo]) -> List[Dict[str, Any]]:
        """Discover possible join paths between tables"""
        
        join_paths = []
        
        for i, table1 in enumerate(tables):
            for j, table2 in enumerate(tables[i+1:], i+1):
                # Look for common column names that could be join keys
                table1_cols = {col.name.lower() for col in table1.columns}
                table2_cols = {col.name.lower() for col in table2.columns}
                
                common_cols = table1_cols.intersection(table2_cols)
                
                # Filter for likely join keys
                join_candidates = []
                for col in common_cols:
                    if any(key_pattern in col for key_pattern in ["id", "key", "code"]):
                        join_candidates.append(col)
                
                if join_candidates:
                    join_path = {
                        "left_table": table1.name,
                        "right_table": table2.name,
                        "join_keys": join_candidates,
                        "join_type": "inner",  # Default, could be smarter
                        "relationship": self._infer_relationship(table1, table2, join_candidates)
                    }
                    join_paths.append(join_path)
        
        return join_paths
    
    async def _map_entities_to_columns(self, entities: List[str], tables: List[TableInfo]) -> Dict[str, List[str]]:
        """Map business entities to specific column names"""
        
        entity_mappings = {}
        
        for entity in entities:
            matching_columns = []
            entity_lower = entity.lower()
            
            for table in tables:
                for column in table.columns:
                    col_lower = column.name.lower()
                    
                    # Direct name match
                    if entity_lower in col_lower or col_lower in entity_lower:
                        matching_columns.append(f"{table.name}.{column.name}")
                    
                    # Business synonym match
                    if column.business_synonyms:
                        for synonym in column.business_synonyms:
                            if entity_lower in synonym.lower() or synonym.lower() in entity_lower:
                                matching_columns.append(f"{table.name}.{column.name}")
                                break
            
            if matching_columns:
                entity_mappings[entity] = matching_columns
        
        return entity_mappings
    
    async def _discover_metrics(self, tables: List[TableInfo], query: str) -> List[Dict[str, Any]]:
        """Discover available metrics and aggregations"""
        
        metrics = []
        query_lower = query.lower()
        
        # Look for aggregation keywords in query
        aggregation_keywords = {
            "count": ["count", "number", "total"],
            "sum": ["sum", "total", "amount"],
            "avg": ["average", "avg", "mean"],
            "max": ["maximum", "max", "highest"],
            "min": ["minimum", "min", "lowest"]
        }
        
        for table in tables:
            for column in table.columns:
                if column.data_type in ["INTEGER", "DECIMAL", "FLOAT", "NUMBER"]:
                    # This could be a measurable metric
                    for agg_type, keywords in aggregation_keywords.items():
                        if any(keyword in query_lower for keyword in keywords):
                            metric = {
                                "name": f"{agg_type}_{column.name}",
                                "table": table.name,
                                "column": column.name,
                                "aggregation": agg_type,
                                "data_type": column.data_type,
                                "description": f"{agg_type.title()} of {column.name}"
                            }
                            metrics.append(metric)
        
        return metrics
    
    async def _discover_date_columns(self, tables: List[TableInfo]) -> List[Dict[str, Any]]:
        """Find date/time columns for temporal filtering"""
        
        date_columns = []
        
        for table in tables:
            for column in table.columns:
                if any(date_type in column.data_type.upper() 
                      for date_type in ["DATE", "TIMESTAMP", "DATETIME"]):
                    
                    date_col = {
                        "table": table.name,
                        "column": column.name,
                        "data_type": column.data_type,
                        "granularity": self._infer_date_granularity(column.name),
                        "is_primary_date": self._is_primary_date_column(column.name)
                    }
                    date_columns.append(date_col)
        
        # Sort by likely importance (primary dates first)
        date_columns.sort(key=lambda x: x["is_primary_date"], reverse=True)
        
        return date_columns
    
    async def _generate_filter_suggestions(self, query: str, tables: List[TableInfo]) -> List[Dict[str, Any]]:
        """Generate smart filter suggestions based on query context"""
        
        suggestions = []
        query_lower = query.lower()
        
        # Time-based filters
        if any(time_word in query_lower for time_word in ["last", "recent", "this", "current"]):
            time_filters = [
                {"type": "date_range", "label": "Last 30 days", "value": "30d"},
                {"type": "date_range", "label": "Last quarter", "value": "1q"},
                {"type": "date_range", "label": "Year to date", "value": "ytd"}
            ]
            suggestions.extend(time_filters)
        
        # Geographic filters
        if any(geo_word in query_lower for geo_word in ["region", "state", "country", "territory"]):
            geo_filters = [
                {"type": "geography", "label": "By Region", "column": "region"},
                {"type": "geography", "label": "By State", "column": "state"}
            ]
            suggestions.extend(geo_filters)
        
        # Product/category filters
        if any(prod_word in query_lower for prod_word in ["product", "category", "segment"]):
            product_filters = [
                {"type": "category", "label": "By Product Category", "column": "category"},
                {"type": "segment", "label": "By Segment", "column": "segment"}
            ]
            suggestions.extend(product_filters)
        
        return suggestions
    
    async def _build_business_glossary(self, entities: List[str]) -> Dict[str, str]:
        """Build business context glossary for entities"""
        
        glossary = {}
        
        # Default business definitions
        default_definitions = {
            "customer": "Individual or organization that purchases products or services",
            "revenue": "Total income generated from sales of products or services",
            "product": "Good or service offered for sale",
            "order": "Customer request to purchase products or services",
            "transaction": "Individual sale or purchase event"
        }
        
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower in default_definitions:
                glossary[entity] = default_definitions[entity_lower]
            else:
                # Generate context-aware definition
                glossary[entity] = f"Business entity: {entity}"
        
        return glossary
    
    def _is_pii_column(self, column_name: str) -> bool:
        """Identify if column likely contains PII"""
        pii_patterns = ["email", "phone", "ssn", "address", "name", "credit_card"]
        col_lower = column_name.lower()
        return any(pattern in col_lower for pattern in pii_patterns)
    
    def _is_key_column(self, column_name: str, column_info: Dict[str, Any]) -> bool:
        """Identify if column is a key column"""
        key_patterns = ["id", "key", "code"]
        col_lower = column_name.lower()
        return any(pattern in col_lower for pattern in key_patterns)
    
    def _get_business_synonyms(self, column_name: str) -> List[str]:
        """Get business synonyms for a column"""
        col_lower = column_name.lower()
        synonyms = []
        
        for business_term, synonym_list in self.business_synonyms.items():
            if business_term in col_lower:
                synonyms.extend(synonym_list)
        
        return synonyms
    
    def _is_fact_table(self, table_name: str, columns: List[ColumnInfo]) -> bool:
        """Determine if table is likely a fact table"""
        table_lower = table_name.lower()
        
        # Check for fact table patterns
        if any(pattern in table_lower for pattern in self.table_patterns["fact"]):
            return True
        
        # Check for measure columns (numeric columns that could be aggregated)
        numeric_cols = [col for col in columns if col.data_type in ["INTEGER", "DECIMAL", "FLOAT", "NUMBER"]]
        
        # Fact tables typically have multiple numeric columns
        return len(numeric_cols) >= 2
    
    def _is_dimension_table(self, table_name: str, columns: List[ColumnInfo]) -> bool:
        """Determine if table is likely a dimension table"""
        table_lower = table_name.lower()
        
        # Check for dimension table patterns
        if any(pattern in table_lower for pattern in self.table_patterns["dimension"]):
            return True
        
        # Dimension tables typically have more text/categorical columns
        text_cols = [col for col in columns if col.data_type in ["VARCHAR", "TEXT", "STRING"]]
        return len(text_cols) > len(columns) * 0.6
    
    def _infer_relationship(self, table1: TableInfo, table2: TableInfo, join_keys: List[str]) -> str:
        """Infer the relationship type between tables"""
        
        # Simple heuristic: if one table is fact and other is dimension
        if table1.is_fact_table and table2.is_dimension_table:
            return "many-to-one"
        elif table1.is_dimension_table and table2.is_fact_table:
            return "one-to-many"
        else:
            return "many-to-many"
    
    def _infer_date_granularity(self, column_name: str) -> str:
        """Infer the date granularity from column name"""
        col_lower = column_name.lower()
        
        if "daily" in col_lower or "day" in col_lower:
            return "daily"
        elif "monthly" in col_lower or "month" in col_lower:
            return "monthly"
        elif "quarterly" in col_lower or "quarter" in col_lower:
            return "quarterly"
        elif "yearly" in col_lower or "year" in col_lower:
            return "yearly"
        else:
            return "daily"  # Default
    
    def _is_primary_date_column(self, column_name: str) -> bool:
        """Determine if this is likely the primary date column"""
        primary_patterns = ["created", "date", "timestamp", "time"]
        col_lower = column_name.lower()
        return any(pattern in col_lower for pattern in primary_patterns)
