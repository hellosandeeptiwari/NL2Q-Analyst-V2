"""
Deterministic Schema Retriever & Join Planner

Role: From natural-language queries, identify relevant columns, map to tables, 
and determine optimal join strategy without generating SQL.

Based on comprehensive matching rules including lexical similarity, 
embeddings, type compatibility, and intelligent join planning.
"""

import re
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

@dataclass
class ColumnMatch:
    column_name: str
    table_name: str
    data_type: str
    score: float
    reasons: List[str]

@dataclass
class TableGroup:
    table_name: str
    table_score: float
    columns: List[str]

@dataclass
class JoinStep:
    left_table: str
    right_table: str
    join_type: str
    on: List[Dict[str, str]]
    rationale: str

@dataclass
class JoinPlan:
    required: bool
    result_grain: str
    steps: List[JoinStep]
    warnings: List[str]

class SchemaRetriever:
    """Deterministic Schema Retriever & Join Planner"""
    
    def __init__(self):
        # Built-in abbreviation expansion rules
        self.abbreviations = {
            'nbrx': ['new_rx', 'new_starts'],
            'trx': ['rx'],
            'hcp': ['provider', 'physician'],
            'dma': ['market', 'region'],
            'qty': ['quantity'],
            'amt': ['amount'],
            'dt': ['date'],
            'wk': ['week'],
            'mo': ['month'],
            'yr': ['year'],
            'nbr': ['number'],
            'cnt': ['count'],
            'vol': ['volume'],
            'pct': ['percent', 'percentage'],
            'avg': ['average'],
            'min': ['minimum'],
            'max': ['maximum']
        }
        
        # Type classification patterns
        self.type_patterns = {
            'time_date': ['date', 'dt', 'day', 'week', 'month', 'year', 'time', 'timestamp'],
            'numeric_measure': ['count', 'qty', 'amount', 'nbrx', 'trx', 'rate', 'score', 'vol', 'value', 'price', 'cost'],
            'identifier': ['id', 'code', 'key', 'nbr'],
            'geo': ['zip', 'state', 'dma', 'region', 'territory', 'county', 'city']
        }
        
        # Common prefixes/suffixes to strip
        self.strip_patterns = ['dim_', 'fact_', '_tbl', '_view', '_id', '_key', '_code', '_nbr']
        
    def normalize_name(self, name: str) -> List[str]:
        """Normalize names: lowercase, split camelCase/snake_case, strip prefixes/suffixes"""
        if not name:
            return []
            
        # Convert to lowercase
        name = name.lower()
        
        # Split camelCase and snake_case
        # Handle camelCase: split on uppercase letters
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
        
        # Split on underscores, spaces, and hyphens
        tokens = re.split(r'[_\s\-]+', name)
        
        # Strip common prefixes/suffixes
        normalized_tokens = []
        for token in tokens:
            if token:
                # Strip known patterns
                for pattern in self.strip_patterns:
                    if token.startswith(pattern):
                        token = token[len(pattern):]
                    elif token.endswith(pattern):
                        token = token[:-len(pattern)]
                
                if token:  # Only add non-empty tokens
                    normalized_tokens.append(token)
        
        return normalized_tokens
    
    def expand_abbreviations(self, tokens: List[str]) -> List[str]:
        """Expand abbreviations using built-in rules"""
        expanded = []
        for token in tokens:
            if token in self.abbreviations:
                expanded.extend(self.abbreviations[token])
            else:
                expanded.append(token)
        return expanded
    
    def classify_type(self, column_name: str, data_type: str) -> str:
        """Classify column type based on name and data type"""
        name_lower = column_name.lower()
        type_lower = data_type.lower()
        
        # Check name patterns
        for type_class, patterns in self.type_patterns.items():
            for pattern in patterns:
                if pattern in name_lower:
                    return type_class
        
        # Check data type
        if any(t in type_lower for t in ['date', 'time', 'timestamp']):
            return 'time_date'
        elif any(t in type_lower for t in ['int', 'float', 'decimal', 'number', 'numeric']):
            return 'numeric_measure'
        elif any(t in type_lower for t in ['varchar', 'char', 'text', 'string']):
            if '_id' in name_lower or name_lower.endswith('id'):
                return 'identifier'
            return 'text'
        
        return 'unknown'
    
    def calculate_lexical_similarity(self, query_tokens: List[str], target_tokens: List[str]) -> float:
        """Calculate fuzzy lexical similarity between token sets"""
        if not query_tokens or not target_tokens:
            return 0.0
        
        # Expand abbreviations for both
        query_expanded = self.expand_abbreviations(query_tokens)
        target_expanded = self.expand_abbreviations(target_tokens)
        
        # Calculate overlap
        query_set = set(query_expanded)
        target_set = set(target_expanded)
        
        intersection = len(query_set & target_set)
        union = len(query_set | target_set)
        
        if union == 0:
            return 0.0
        
        # Jaccard similarity with bonus for exact matches
        jaccard = intersection / union
        
        # Add bonus for exact token matches
        exact_matches = sum(1 for qt in query_tokens if qt in target_tokens)
        exact_bonus = exact_matches / max(len(query_tokens), len(target_tokens))
        
        return min(1.0, jaccard + 0.2 * exact_bonus)
    
    def calculate_embedding_similarity(self, query: str, target: str) -> float:
        """Calculate cosine similarity between embeddings (simplified version)"""
        # For now, use a simplified approach based on character-level similarity
        # In production, you'd use actual embeddings
        
        if not query or not target:
            return 0.0
        
        # Simple character-based similarity as placeholder
        query_chars = set(query.lower())
        target_chars = set(target.lower())
        
        intersection = len(query_chars & target_chars)
        union = len(query_chars | target_chars)
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_type_compatibility(self, query_tokens: List[str], column_type: str) -> float:
        """Calculate type compatibility score"""
        query_text = ' '.join(query_tokens).lower()
        
        type_indicators = {
            'time_date': ['date', 'time', 'day', 'week', 'month', 'year', 'when', 'during'],
            'numeric_measure': ['count', 'amount', 'total', 'sum', 'average', 'number', 'quantity'],
            'identifier': ['id', 'code', 'key', 'identifier'],
            'geo': ['location', 'address', 'state', 'zip', 'region', 'territory']
        }
        
        if column_type in type_indicators:
            indicators = type_indicators[column_type]
            matches = sum(1 for indicator in indicators if indicator in query_text)
            return min(1.0, matches / len(indicators))
        
        return 0.0
    
    def score_column(self, query: str, query_tokens: List[str], column: Dict[str, str], 
                    table_name_boost: float = 0.0) -> ColumnMatch:
        """Score a column based on the comprehensive scoring formula"""
        
        column_name = column['column_name']
        table_name = column['table_name']
        data_type = column['data_type']
        
        # Normalize column and table names
        column_tokens = self.normalize_name(column_name)
        table_tokens = self.normalize_name(table_name)
        
        # Calculate similarity components
        embed_sim = self.calculate_embedding_similarity(query, column_name)
        lexical_sim = self.calculate_lexical_similarity(query_tokens, column_tokens)
        
        column_type = self.classify_type(column_name, data_type)
        type_match = self.calculate_type_compatibility(query_tokens, column_type)
        
        # Table name boost (if query tokens overlap with table name)
        table_boost = 0.0
        if table_name_boost > 0:
            table_lexical = self.calculate_lexical_similarity(query_tokens, table_tokens)
            table_boost = table_lexical * table_name_boost
        
        # Combined score: 0.55*Embed + 0.25*Lexical + 0.15*TypeMatch + 0.05*TableNameBoost
        score = (0.55 * embed_sim + 
                0.25 * lexical_sim + 
                0.15 * type_match + 
                0.05 * table_boost)
        
        # Generate reasons for scoring
        reasons = []
        if embed_sim > 0.3:
            reasons.append(f"semantic match ({embed_sim:.2f})")
        if lexical_sim > 0.3:
            reasons.append(f"lexical match ({lexical_sim:.2f})")
        if type_match > 0.2:
            reasons.append(f"type compatible ({column_type})")
        if table_boost > 0.1:
            reasons.append("table name relevance")
        
        return ColumnMatch(
            column_name=column_name,
            table_name=table_name,
            data_type=data_type,
            score=score,
            reasons=reasons
        )
    
    def group_by_tables(self, selected_columns: List[ColumnMatch]) -> List[TableGroup]:
        """Group selected columns by table and calculate table scores"""
        
        table_groups = defaultdict(list)
        for col in selected_columns:
            table_groups[col.table_name].append(col)
        
        # Calculate table scores
        max_columns_in_table = max(len(cols) for cols in table_groups.values()) if table_groups else 1
        
        table_results = []
        for table_name, columns in table_groups.items():
            # Average score of selected columns
            avg_score = sum(col.score for col in columns) / len(columns)
            
            # Coverage bonus
            coverage_bonus = len(columns) / max_columns_in_table
            table_score = avg_score + 0.1 * coverage_bonus
            
            table_results.append(TableGroup(
                table_name=table_name,
                table_score=table_score,
                columns=[col.column_name for col in columns]
            ))
        
        return sorted(table_results, key=lambda x: x.table_score, reverse=True)
    
    def find_join_keys(self, table1: str, table2: str, all_columns: List[Dict]) -> List[Dict[str, str]]:
        """Find potential join keys between two tables"""
        
        table1_cols = [col for col in all_columns if col['table_name'] == table1]
        table2_cols = [col for col in all_columns if col['table_name'] == table2]
        
        join_candidates = []
        
        # Look for identical column names (especially _id columns)
        for col1 in table1_cols:
            for col2 in table2_cols:
                if col1['column_name'] == col2['column_name']:
                    # Prefer _id columns
                    priority = 10 if col1['column_name'].endswith('_id') else 5
                    join_candidates.append({
                        'left': f"{table1}.{col1['column_name']}",
                        'right': f"{table2}.{col2['column_name']}",
                        'priority': priority
                    })
        
        # Look for natural key matches (same base name, compatible types)
        if not join_candidates:
            for col1 in table1_cols:
                col1_tokens = self.normalize_name(col1['column_name'])
                for col2 in table2_cols:
                    col2_tokens = self.normalize_name(col2['column_name'])
                    
                    # Check for token overlap
                    overlap = set(col1_tokens) & set(col2_tokens)
                    if overlap and len(overlap) >= 1:
                        join_candidates.append({
                            'left': f"{table1}.{col1['column_name']}",
                            'right': f"{table2}.{col2['column_name']}",
                            'priority': len(overlap)
                        })
        
        # Return best candidates
        join_candidates.sort(key=lambda x: x['priority'], reverse=True)
        return [{'left': jc['left'], 'right': jc['right']} for jc in join_candidates[:2]]
    
    def determine_join_plan(self, selected_tables: List[TableGroup], 
                          all_columns: List[Dict], query: str) -> JoinPlan:
        """Determine if joins are required and create join plan"""
        
        if len(selected_tables) <= 1:
            # Single table - no join required
            if selected_tables:
                # Find key columns for grain
                table_cols = [col for col in all_columns 
                            if col['table_name'] == selected_tables[0].table_name]
                key_cols = [col['column_name'] for col in table_cols 
                           if col['column_name'].endswith('_id') or 'key' in col['column_name'].lower()]
                
                grain = ', '.join(key_cols[:2]) if key_cols else "single table"
            else:
                grain = "unknown"
                
            return JoinPlan(
                required=False,
                result_grain=grain,
                steps=[],
                warnings=[]
            )
        
        # Check if query requires combining data on same row
        combining_keywords = ['with', 'and', 'by', 'per', 'top', 'compare', 'correlation']
        needs_join = any(keyword in query.lower() for keyword in combining_keywords)
        
        if not needs_join:
            # Multiple tables but separate facts
            return JoinPlan(
                required=False,
                result_grain="per-table",
                steps=[],
                warnings=["Multiple tables selected but no join indicated"]
            )
        
        # Build join plan
        join_steps = []
        warnings = []
        
        # Identify fact vs dimension tables (tables with many numeric measures are facts)
        fact_tables = []
        dim_tables = []
        
        for table_group in selected_tables:
            table_cols = [col for col in all_columns if col['table_name'] == table_group.table_name]
            numeric_cols = sum(1 for col in table_cols 
                             if self.classify_type(col['column_name'], col['data_type']) == 'numeric_measure')
            
            if numeric_cols >= 2:
                fact_tables.append(table_group)
            else:
                dim_tables.append(table_group)
        
        # If no clear fact table, treat largest table as fact
        if not fact_tables:
            largest_table = max(selected_tables, key=lambda x: len(x.columns))
            fact_tables = [largest_table]
            dim_tables = [t for t in selected_tables if t != largest_table]
        
        # Create joins from fact to dimensions
        primary_fact = fact_tables[0]
        
        for dim_table in dim_tables:
            join_keys = self.find_join_keys(primary_fact.table_name, dim_table.table_name, all_columns)
            
            if join_keys:
                join_steps.append(JoinStep(
                    left_table=primary_fact.table_name,
                    right_table=dim_table.table_name,
                    join_type="LEFT",
                    on=join_keys[:1],  # Use best join key
                    rationale=f"Join fact to dimension on {join_keys[0]['left'].split('.')[1]}"
                ))
            else:
                warnings.append(f"No clear join path found between {primary_fact.table_name} and {dim_table.table_name}")
        
        # Add additional fact tables if any
        for additional_fact in fact_tables[1:]:
            join_keys = self.find_join_keys(primary_fact.table_name, additional_fact.table_name, all_columns)
            
            if join_keys:
                join_steps.append(JoinStep(
                    left_table=primary_fact.table_name,
                    right_table=additional_fact.table_name,
                    join_type="INNER",
                    on=join_keys[:1],
                    rationale=f"Join facts on {join_keys[0]['left'].split('.')[1]}"
                ))
        
        # Determine result grain
        key_cols = []
        for table_group in selected_tables:
            table_cols = [col for col in all_columns if col['table_name'] == table_group.table_name]
            table_keys = [col['column_name'] for col in table_cols 
                         if col['column_name'].endswith('_id') or 'key' in col['column_name'].lower()]
            key_cols.extend(table_keys)
        
        grain = ', '.join(list(set(key_cols))[:3]) if key_cols else "joined tables"
        
        return JoinPlan(
            required=True,
            result_grain=grain,
            steps=join_steps,
            warnings=warnings
        )
    
    def retrieve_schema(self, user_question: str, catalog: Dict[str, Any], 
                       top_k_columns: int = 12) -> Dict[str, Any]:
        """Main entry point: retrieve schema and plan joins from natural language query"""
        
        # Extract inputs
        tables = catalog.get('tables', [])
        columns = catalog.get('columns', [])
        
        # Normalize and tokenize query
        query_tokens = self.normalize_name(user_question)
        query_expanded = self.expand_abbreviations(query_tokens)
        
        # Calculate table name boosts
        table_boosts = {}
        for table in tables:
            table_tokens = self.normalize_name(table['name'])
            table_overlap = self.calculate_lexical_similarity(query_expanded, table_tokens)
            table_boosts[table['name']] = table_overlap
        
        # Score all columns
        scored_columns = []
        for column in columns:
            table_boost = table_boosts.get(column['table_name'], 0.0)
            column_match = self.score_column(user_question, query_expanded, column, table_boost)
            scored_columns.append(column_match)
        
        # Apply threshold and select top columns
        threshold = 0.52
        qualified_columns = [col for col in scored_columns if col.score >= threshold]
        
        # Relaxed threshold if too few candidates
        if len(qualified_columns) < 3:
            threshold = 0.48
            qualified_columns = [col for col in scored_columns if col.score >= threshold]
        
        # Take top_k columns
        qualified_columns.sort(key=lambda x: x.score, reverse=True)
        selected_columns = qualified_columns[:top_k_columns]
        
        # Group by tables
        table_groups = self.group_by_tables(selected_columns)
        
        # Determine join plan
        join_plan = self.determine_join_plan(table_groups, columns, user_question)
        
        # Identify unmatched intents
        unmatched_terms = []
        for token in query_expanded:
            found = any(token in col.column_name.lower() or 
                       any(token in reason.lower() for reason in col.reasons)
                       for col in selected_columns)
            if not found:
                unmatched_terms.append(token)
        
        # Calculate overall confidence
        avg_score = sum(col.score for col in selected_columns) / len(selected_columns) if selected_columns else 0.0
        coverage_factor = min(1.0, len(selected_columns) / max(6, len(query_expanded)))
        confidence_overall = avg_score * coverage_factor
        
        # Format output
        return {
            "columns_selected": [
                {
                    "column_name": col.column_name,
                    "table_name": col.table_name,
                    "data_type": col.data_type,
                    "score": round(col.score, 3),
                    "reasons": col.reasons
                }
                for col in selected_columns
            ],
            "tables_selected": [
                {
                    "table_name": tg.table_name,
                    "table_score": round(tg.table_score, 3),
                    "columns": tg.columns
                }
                for tg in table_groups
            ],
            "join_plan": {
                "required": join_plan.required,
                "result_grain": join_plan.result_grain,
                "steps": [
                    {
                        "left_table": step.left_table,
                        "right_table": step.right_table,
                        "join_type": step.join_type,
                        "on": step.on,
                        "rationale": step.rationale
                    }
                    for step in join_plan.steps
                ],
                "warnings": join_plan.warnings
            },
            "unmatched_intents": unmatched_terms,
            "confidence_overall": round(confidence_overall, 3)
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the schema retriever
    retriever = SchemaRetriever()
    
    # Example catalog
    catalog = {
        "tables": [
            {"name": "FACT_NBRX_WEEKLY"},
            {"name": "DIM_HCP_ADDRESS"},
            {"name": "ADMIN_METRICS"},
            {"name": "BILLING_PRICES"}
        ],
        "columns": [
            {"table_name": "FACT_NBRX_WEEKLY", "column_name": "hcp_id", "data_type": "VARCHAR"},
            {"table_name": "FACT_NBRX_WEEKLY", "column_name": "nbrx_count", "data_type": "INTEGER"},
            {"table_name": "FACT_NBRX_WEEKLY", "column_name": "week_dt", "data_type": "DATE"},
            {"table_name": "DIM_HCP_ADDRESS", "column_name": "hcp_id", "data_type": "VARCHAR"},
            {"table_name": "DIM_HCP_ADDRESS", "column_name": "state", "data_type": "VARCHAR"},
            {"table_name": "ADMIN_METRICS", "column_name": "warehouse_size", "data_type": "INTEGER"},
            {"table_name": "BILLING_PRICES", "column_name": "credit_price", "data_type": "DECIMAL"}
        ]
    }
    
    # Test cases
    test_queries = [
        "Show HCP, NBRx, and Week",  # Single table
        "What is warehouse size and credit price?",  # Multi-table, no join
        "Top 10 HCPs by NBRx with their state"  # Join required
    ]
    
    for query in test_queries:
        print(f"\n=== Query: {query} ===")
        result = retriever.retrieve_schema(query, catalog)
        print(json.dumps(result, indent=2))
