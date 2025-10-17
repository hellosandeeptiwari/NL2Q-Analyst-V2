"""
Dynamic Filter Value Resolver
Automatically resolves filter values by querying the database for actual values.
No hardcoding - adapts to any database schema dynamically.

Examples:
- User says "PDRP enabled" → Finds actual values (YES/NO or Y/N or 1/0)
- User says "John Smith" → Finds case-sensitive match or fuzzy matches
- User says "last quarter" → Resolves to actual date range
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
from difflib import SequenceMatcher
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class FilterMatch:
    """Represents a matched filter value"""
    column_name: str
    user_value: str
    actual_value: str
    match_type: str  # 'exact', 'fuzzy', 'boolean', 'enum'
    confidence: float


@dataclass
class ColumnValues:
    """Cache of column values from database"""
    column_name: str
    distinct_values: List[Any]
    cardinality: int
    data_type: str
    is_boolean: bool = False
    is_enum: bool = False


class FilterValueResolver:
    """
    Dynamically resolves filter values by querying actual database values.
    No hardcoding - learns from the database itself.
    """
    
    def __init__(self, db_adapter):
        self.db_adapter = db_adapter
        self.value_cache: Dict[str, ColumnValues] = {}
        
        # Generic boolean mappings (detected dynamically)
        self.boolean_patterns = {
            'true': ['yes', 'y', 'true', 't', '1', 'enabled', 'active', 'on'],
            'false': ['no', 'n', 'false', 'f', '0', 'disabled', 'inactive', 'off']
        }
    
    def resolve_filter_values(
        self, 
        query_text: str, 
        schema_context: Dict[str, Any]
    ) -> Dict[str, List[FilterMatch]]:
        """
        Main method: Extract filters from user query and resolve to actual DB values.
        
        Args:
            query_text: User's natural language query
            schema_context: Schema information including table and column names
            
        Returns:
            Dict mapping column names to resolved filter values
        """
        logger.info(f"🔍 Resolving filter values in query: {query_text[:100]}...")
        
        filters = self._extract_potential_filters(query_text)
        resolved_filters = {}
        
        for filter_hint in filters:
            matches = self._resolve_filter_hint(filter_hint, schema_context)
            if matches:
                for match in matches:
                    if match.column_name not in resolved_filters:
                        resolved_filters[match.column_name] = []
                    resolved_filters[match.column_name].append(match)
        
        return resolved_filters
    
    def _extract_potential_filters(self, query_text: str) -> List[Dict[str, str]]:
        """
        Extract potential filter conditions from natural language.
        
        Examples:
        - "PDRP enabled" → {column: "PDRP", value: "enabled", operator: "="}
        - "status is active" → {column: "status", value: "active", operator: "="}
        - "John Smith" → {column: None, value: "John Smith", operator: "="}
        """
        filters = []
        
        # Pattern 1: "column_name = value" or "column_name is value"
        pattern1 = r'(\w+)\s+(?:is|=|equals?)\s+["\']?(\w+)["\']?'
        for match in re.finditer(pattern1, query_text, re.IGNORECASE):
            filters.append({
                'column': match.group(1),
                'value': match.group(2),
                'operator': '='
            })
        
        # Pattern 2: "value column_name" (e.g., "enabled PDRP", "active status")
        pattern2 = r'(\w+)\s+(flag|status|type|category|state|indicator)\b'
        for match in re.finditer(pattern2, query_text, re.IGNORECASE):
            filters.append({
                'column': match.group(2),
                'value': match.group(1),
                'operator': '='
            })
        
        # Pattern 3: Boolean flags (enabled, disabled, active, inactive)
        boolean_keywords = ['enabled', 'disabled', 'active', 'inactive', 'yes', 'no']
        for keyword in boolean_keywords:
            if keyword in query_text.lower():
                # Try to find nearby column names
                pattern = rf'(\w*(?:flag|status|target|indicator)\w*)\s+{keyword}|{keyword}\s+(\w*(?:flag|status|target|indicator)\w*)'
                for match in re.finditer(pattern, query_text, re.IGNORECASE):
                    column = match.group(1) or match.group(2)
                    if column:
                        filters.append({
                            'column': column,
                            'value': keyword,
                            'operator': '='
                        })
        
        # Pattern 4: Quoted values (exact name matches)
        pattern4 = r'["\']([^"\']+)["\']'
        for match in re.finditer(pattern4, query_text):
            filters.append({
                'column': None,  # Will search all text columns
                'value': match.group(1),
                'operator': '='
            })
        
        logger.info(f"📋 Extracted {len(filters)} potential filters: {filters}")
        return filters
    
    def _resolve_filter_hint(
        self, 
        filter_hint: Dict[str, str], 
        schema_context: Dict[str, Any]
    ) -> List[FilterMatch]:
        """Resolve a single filter hint to actual database values"""
        
        column_name = filter_hint.get('column')
        user_value = filter_hint.get('value')
        
        if not user_value:
            return []
        
        matches = []
        
        # If column specified, resolve for that column
        if column_name:
            column_matches = self._find_column_fuzzy(column_name, schema_context)
            for col_match in column_matches:
                actual_col_name = col_match['column_name']
                table_name = col_match['table_name']
                
                # Get actual values from database
                col_values = self._get_column_values(table_name, actual_col_name)
                if col_values:
                    value_match = self._match_value(user_value, col_values)
                    if value_match:
                        matches.append(value_match)
        
        # If no column specified, search all text/enum columns
        else:
            for table_info in schema_context.get('tables', []):
                table_name = table_info.get('name')
                for column_info in table_info.get('columns', []):
                    col_name = column_info.get('name')
                    data_type = column_info.get('type', '').lower()
                    
                    # Only search text columns for name matches
                    if any(t in data_type for t in ['varchar', 'char', 'text', 'string']):
                        col_values = self._get_column_values(table_name, col_name)
                        if col_values:
                            value_match = self._match_value(user_value, col_values)
                            if value_match and value_match.confidence > 0.8:
                                matches.append(value_match)
        
        return matches
    
    def _find_column_fuzzy(
        self, 
        column_hint: str, 
        schema_context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Find columns matching the hint using fuzzy matching"""
        
        matches = []
        column_hint_lower = column_hint.lower()
        
        for table_info in schema_context.get('tables', []):
            table_name = table_info.get('name')
            for column_info in table_info.get('columns', []):
                col_name = column_info.get('name')
                col_name_lower = col_name.lower()
                
                # Exact match
                if column_hint_lower == col_name_lower:
                    matches.append({
                        'column_name': col_name,
                        'table_name': table_name,
                        'confidence': 1.0
                    })
                
                # Partial match (hint is part of column name)
                elif column_hint_lower in col_name_lower:
                    matches.append({
                        'column_name': col_name,
                        'table_name': table_name,
                        'confidence': 0.8
                    })
                
                # Fuzzy match (similar strings)
                else:
                    ratio = SequenceMatcher(None, column_hint_lower, col_name_lower).ratio()
                    if ratio > 0.7:
                        matches.append({
                            'column_name': col_name,
                            'table_name': table_name,
                            'confidence': ratio
                        })
        
        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches[:3]  # Return top 3 matches
    
    def _get_column_values(
        self, 
        table_name: str, 
        column_name: str,
        limit: int = 100
    ) -> Optional[ColumnValues]:
        """
        Get distinct values for a column from the database.
        Cached for performance.
        """
        
        cache_key = f"{table_name}.{column_name}"
        
        # Check cache
        if cache_key in self.value_cache:
            return self.value_cache[cache_key]
        
        try:
            # Query for distinct values and cardinality
            query = f"""
            SELECT TOP {limit}
                [{column_name}],
                COUNT(*) as frequency
            FROM {table_name}
            WHERE [{column_name}] IS NOT NULL
            GROUP BY [{column_name}]
            ORDER BY COUNT(*) DESC
            """
            
            result = self.db_adapter.run(query)
            
            if result.error:
                logger.warning(f"⚠️ Could not fetch values for {table_name}.{column_name}: {result.error}")
                return None
            
            distinct_values = [row[0] for row in result.rows]
            cardinality = len(distinct_values)
            
            # Detect data type and characteristics
            if distinct_values:
                sample_value = distinct_values[0]
                data_type = type(sample_value).__name__
                
                # Detect boolean column
                is_boolean = self._is_boolean_column(distinct_values)
                
                # Detect enum column (low cardinality)
                is_enum = cardinality <= 50
                
                col_values = ColumnValues(
                    column_name=column_name,
                    distinct_values=distinct_values,
                    cardinality=cardinality,
                    data_type=data_type,
                    is_boolean=is_boolean,
                    is_enum=is_enum
                )
                
                # Cache it
                self.value_cache[cache_key] = col_values
                
                logger.info(f"✅ Cached {cardinality} values for {cache_key} (boolean={is_boolean}, enum={is_enum})")
                return col_values
        
        except Exception as e:
            logger.error(f"❌ Error fetching column values: {e}")
            return None
        
        return None
    
    def _is_boolean_column(self, values: List[Any]) -> bool:
        """Detect if column contains boolean-like values"""
        
        if not values or len(values) > 5:
            return False
        
        # Convert to lowercase strings
        str_values = [str(v).lower() for v in values]
        
        # Check against boolean patterns
        true_values = set(self.boolean_patterns['true'])
        false_values = set(self.boolean_patterns['false'])
        
        matched = set(str_values) & (true_values | false_values)
        
        return len(matched) == len(str_values)
    
    def _match_value(
        self, 
        user_value: str, 
        col_values: ColumnValues
    ) -> Optional[FilterMatch]:
        """
        Match user's value to actual database value.
        Handles fuzzy matching, case differences, boolean mapping.
        """
        
        user_value_lower = str(user_value).lower()
        
        # Boolean column - map user intent to actual values
        if col_values.is_boolean:
            return self._match_boolean_value(user_value_lower, col_values)
        
        # Exact match (case-insensitive)
        for db_value in col_values.distinct_values:
            if str(db_value).lower() == user_value_lower:
                return FilterMatch(
                    column_name=col_values.column_name,
                    user_value=user_value,
                    actual_value=str(db_value),
                    match_type='exact',
                    confidence=1.0
                )
        
        # Fuzzy match for enums
        if col_values.is_enum:
            best_match = None
            best_ratio = 0.0
            
            for db_value in col_values.distinct_values:
                db_value_str = str(db_value).lower()
                
                # Partial match
                if user_value_lower in db_value_str or db_value_str in user_value_lower:
                    ratio = 0.85
                else:
                    # Sequence matching
                    ratio = SequenceMatcher(None, user_value_lower, db_value_str).ratio()
                
                if ratio > best_ratio and ratio > 0.7:
                    best_ratio = ratio
                    best_match = db_value
            
            if best_match:
                return FilterMatch(
                    column_name=col_values.column_name,
                    user_value=user_value,
                    actual_value=str(best_match),
                    match_type='fuzzy',
                    confidence=best_ratio
                )
        
        return None
    
    def _match_boolean_value(
        self, 
        user_value: str, 
        col_values: ColumnValues
    ) -> Optional[FilterMatch]:
        """Match boolean-like user input to actual database boolean values"""
        
        # Determine if user wants true or false
        is_true_intent = user_value in self.boolean_patterns['true']
        is_false_intent = user_value in self.boolean_patterns['false']
        
        if not (is_true_intent or is_false_intent):
            return None
        
        # Find the actual true/false values in database
        db_true_value = None
        db_false_value = None
        
        for db_value in col_values.distinct_values:
            db_value_lower = str(db_value).lower()
            
            if db_value_lower in self.boolean_patterns['true']:
                db_true_value = db_value
            elif db_value_lower in self.boolean_patterns['false']:
                db_false_value = db_value
        
        # Return appropriate match
        if is_true_intent and db_true_value:
            return FilterMatch(
                column_name=col_values.column_name,
                user_value=user_value,
                actual_value=str(db_true_value),
                match_type='boolean',
                confidence=1.0
            )
        elif is_false_intent and db_false_value:
            return FilterMatch(
                column_name=col_values.column_name,
                user_value=user_value,
                actual_value=str(db_false_value),
                match_type='boolean',
                confidence=1.0
            )
        
        return None
    
    def generate_filter_guidance(
        self, 
        resolved_filters: Dict[str, List[FilterMatch]]
    ) -> str:
        """
        Generate prompt guidance for SQL generation based on resolved filters.
        """
        
        if not resolved_filters:
            return ""
        
        guidance_parts = [
            "\n🎯 DETECTED FILTER VALUES (use these exact values in WHERE clause):"
        ]
        
        for column_name, matches in resolved_filters.items():
            for match in matches:
                if match.confidence >= 0.8:
                    guidance_parts.append(
                        f"  • {column_name} = '{match.actual_value}' "
                        f"(user said '{match.user_value}', confidence={match.confidence:.0%})"
                    )
        
        guidance_parts.append("\n⚠️  Use the ACTUAL VALUES shown above, not the user's original terms!")
        
        return "\n".join(guidance_parts)


# Singleton instance
_resolver_instance = None


def get_filter_resolver(db_adapter) -> FilterValueResolver:
    """Get or create filter resolver singleton"""
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = FilterValueResolver(db_adapter)
    return _resolver_instance
