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
        
        # Pattern 5: Product name patterns - COMPREHENSIVE COVERAGE
        # Examples: "for Tirosint", "Tirosint Capsules data", "related to Tirosint", "about Tirosint"
        #           "Tirosint product", "from Tirosint family", "of Tirosint brand"
        product_patterns = [
            # "for <product>" - most common
            r'\bfor\s+([\w\s]+?)(?:\s+during|\s+in|\s+from|\s+between|\s+across|$)',
            # "of <product>", "about <product>", "regarding <product>"
            r'\b(?:of|about|regarding|concerning)\s+([\w\s]+?)(?:\s+during|\s+in|\s+from|\s+product|\s+brand|$)',
            # "related to <product>"
            r'\brelated\s+to\s+([\w\s]+?)(?:\s+during|\s+in|\s+from|\s+product|$)',
            # "<product> product/brand/line"
            r'\b([\w\s]+?)\s+(?:product|brand|line|family|group)(?:\s+during|\s+in|\s+from|$)',
            # "from <product> line/family"
            r'\bfrom\s+(?:the\s+)?([\w\s]+?)\s+(?:product|brand|line|family|group)',
        ]
        
        for pattern in product_patterns:
            for match in re.finditer(pattern, query_text, re.IGNORECASE):
                product_name = match.group(1).strip()
                # Filter out common stop words and SQL keywords
                stop_words = ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'all', 'any', 'each', 
                             'every', 'some', 'many', 'few', 'most', 'several', 'total', 'overall']
                if product_name.lower() not in stop_words and len(product_name) > 2:
                    filters.append({
                        'column': None,
                        'value': product_name,
                        'operator': '=',
                        'hint': 'product'
                    })
        
        # Pattern 5b: Territory/Region/State patterns
        # Examples: "in California", "from New York", "across Texas", "for territory T123"
        location_patterns = [
            r'\b(?:in|from|across|for|within)\s+(?:territory|region|state|area)\s+([\w\s]+?)(?:\s+during|\s+in|\s+from|$)',
            r'\b(?:in|from|across|within)\s+(California|Texas|New York|Florida|Illinois|[\w\s]+\s+state)(?:\s+during|\s+in|$)',
            r'\bterritory\s+(\w+)',
            r'\bregion\s+(\w+)',
        ]
        
        for pattern in location_patterns:
            for match in re.finditer(pattern, query_text, re.IGNORECASE):
                location = match.group(1).strip()
                if len(location) > 1:
                    filters.append({
                        'column': None,
                        'value': location,
                        'operator': '=',
                        'hint': 'location'
                    })
        
        # Pattern 5c: Prescriber/Account patterns
        # Examples: "prescriber ID 12345", "account ABC123", "NPI 1234567890"
        prescriber_patterns = [
            r'\b(?:prescriber|doctor|physician)\s+(?:ID|id)?\s*[:#]?\s*(\w+)',
            r'\baccount\s+(?:ID|id|name)?\s*[:#]?\s*(\w+)',
            r'\bNPI\s+(?:number)?\s*[:#]?\s*(\d+)',
        ]
        
        for pattern in prescriber_patterns:
            for match in re.finditer(pattern, query_text, re.IGNORECASE):
                identifier = match.group(1).strip()
                filters.append({
                    'column': None,
                    'value': identifier,
                    'operator': '=',
                    'hint': 'prescriber'
                })
        
        # Pattern 6: Time periods - COMPREHENSIVE PATTERN COVERAGE
        
        # Quarter patterns - handles multiple variations
        # Examples: "Q2 2022", "Q2 of 2022", "Q2 for 2022", "Q2 in 2022", "in Q2 2022", "during Q2 2022"
        #           "2022-Q2", "2022 Q2", "second quarter of 2022", "2nd quarter 2022", "second quarter 2022"
        quarter_patterns = [
            # Standard formats: Q2 2022, Q2 of 2022, Q2 for 2022, Q2 in 2022
            r'\b(Q[1-4])\s+(?:of|for|in|during)?\s*(\d{4})\b',
            # Reverse formats: 2022-Q2, 2022 Q2
            r'\b(\d{4})[-/\s](Q[1-4])\b',
            # With prepositions: in Q2 2022, during Q2 of 2022, for Q2 2022
            r'\b(?:in|during|for)\s+(Q[1-4])\s+(?:of|for|in)?\s*(\d{4})\b',
            # Written out with optional prepositions: "first quarter 2022", "second quarter of 2022", "third quarter for 2022", "second quarter 2022"
            r'\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+(?:of|for|in)?\s*(\d{4})\b',
            # With prepositions before written quarter: "in second quarter 2022", "during first quarter of 2022"
            r'\b(?:in|during|for)\s+(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+(?:of|for|in)?\s*(\d{4})\b',
            # Quarter abbreviations: "Q2 FY2022", "1st Qtr 2022", "2Q 2022"
            r'\b([1-4])Q\s+(\d{4})\b',  # 2Q 2022
            r'\b(Q[1-4])\s+FY\s*(\d{4})\b',  # Q2 FY2022
            r'\b([1-4])(?:st|nd|rd|th)\s+(?:qtr|quarter)\s+(?:of|for|in)?\s*(\d{4})\b',  # 1st Qtr 2022
        ]
        
        for pattern in quarter_patterns:
            for match in re.finditer(pattern, query_text, re.IGNORECASE):
                if 'quarter' in pattern.lower() or 'qtr' in pattern.lower():
                    # Handle written out quarters
                    quarter_map = {
                        'first': 'Q1', '1st': 'Q1', '1': 'Q1',
                        'second': 'Q2', '2nd': 'Q2', '2': 'Q2',
                        'third': 'Q3', '3rd': 'Q3', '3': 'Q3',
                        'fourth': 'Q4', '4th': 'Q4', '4': 'Q4'
                    }
                    quarter_text = match.group(1).lower()
                    quarter = quarter_map.get(quarter_text, 'Q1')
                    year = match.group(2)
                    time_value = f"{quarter} {year}"
                elif match.group(1) and match.group(1).startswith('Q'):
                    # Q2 2022 format
                    time_value = f"{match.group(1)} {match.group(2)}"
                elif match.group(1) and match.group(1).isdigit() and len(match.group(1)) == 1:
                    # 2Q 2022 format or single digit quarter
                    quarter_map = {'1': 'Q1', '2': 'Q2', '3': 'Q3', '4': 'Q4'}
                    quarter = quarter_map.get(match.group(1), 'Q1')
                    time_value = f"{quarter} {match.group(2)}"
                elif match.group(1) and match.group(1).isdigit() and len(match.group(1)) == 4:
                    # 2022-Q2 format (year first)
                    time_value = f"{match.group(2)} {match.group(1)}"
                else:
                    # in Q2 2022 format or other formats
                    time_value = f"{match.group(1)} {match.group(2)}"
                
                filters.append({
                    'column': None,
                    'value': time_value,
                    'operator': '=',
                    'hint': 'time_period'
                })
        
        # Additional quarter pattern for edge cases: "the second quarter 2022", "a first quarter 2022"
        # This catches patterns that might be missed with articles
        additional_quarter_pattern = r'\b(?:the|a|an)?\s*(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+(?:of|for|in)?\s*(\d{4})\b'
        for match in re.finditer(additional_quarter_pattern, query_text, re.IGNORECASE):
            quarter_map = {
                'first': 'Q1', '1st': 'Q1',
                'second': 'Q2', '2nd': 'Q2',
                'third': 'Q3', '3rd': 'Q3',
                'fourth': 'Q4', '4th': 'Q4'
            }
            quarter_text = match.group(1).lower()
            quarter = quarter_map.get(quarter_text, 'Q1')
            year = match.group(2)
            time_value = f"{quarter} {year}"
            filters.append({
                'column': None,
                'value': time_value,
                'operator': '=',
                'hint': 'time_period'
            })
        
        # Month/Year patterns - enhanced with prepositions
        # Examples: "January 2022", "in January 2022", "during January of 2022", "for January 2022"
        month_patterns = [
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:of\s+)?(\d{4})\b',
            r'\b(?:in|during|for)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:of\s+)?(\d{4})\b',
            # Abbreviated months: Jan 2022, Feb 2022, etc.
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(?:of\s+)?(\d{4})\b',
        ]
        
        for pattern in month_patterns:
            for match in re.finditer(pattern, query_text, re.IGNORECASE):
                month = match.group(1)
                year = match.group(2)
                time_value = f"{month} {year}"
                filters.append({
                    'column': None,
                    'value': time_value,
                    'operator': '=',
                    'hint': 'time_period'
                })
        
        # Week patterns
        # Examples: "week 12 2022", "week 12 of 2022", "W12 2022", "in week 12 2022"
        week_patterns = [
            r'\b(?:week|W|wk)\s*(\d{1,2})\s+(?:of|in|for)?\s*(\d{4})\b',
            r'\b(?:in|during|for)\s+(?:week|W|wk)\s*(\d{1,2})\s+(?:of)?\s*(\d{4})\b',
        ]
        
        for pattern in week_patterns:
            for match in re.finditer(pattern, query_text, re.IGNORECASE):
                week = match.group(1)
                year = match.group(2)
                time_value = f"W{week} {year}"
                filters.append({
                    'column': None,
                    'value': time_value,
                    'operator': '=',
                    'hint': 'time_period'
                })
        
        # Year patterns - enhanced
        # Examples: "in 2022", "during 2022", "for 2022", "for the year 2022", "FY2022", "fiscal year 2022"
        year_patterns = [
            r'\b(?:in|during|for)\s+(?:the\s+year\s+)?(\d{4})\b',
            r'\b(?:FY|fiscal\s+year)\s*(\d{4})\b',
            r'\bYTD\s+(\d{4})\b',  # Year to date
        ]
        
        for pattern in year_patterns:
            for match in re.finditer(pattern, query_text, re.IGNORECASE):
                year_value = match.group(1)
                filters.append({
                    'column': None,
                    'value': year_value,
                    'operator': '=',
                    'hint': 'time_period'
                })
        
        # NOTE: We do NOT extract relative time periods like "current week", "last 4 weeks"
        # because they cannot be fuzzy-matched to database abbreviations like "C WK", "C4 WK"
        # Instead, the LLM sees the actual TimePeriod values in the schema context and
        # intelligently maps user phrases to the correct database values.
        
        # Pattern 7: Date ranges
        # Examples: "between Q1 2022 and Q3 2022", "from January to March", "Q1-Q3 2022"
        range_patterns = [
            r'\bbetween\s+([\w\s]+?)\s+and\s+([\w\s]+?)(?:\s+in|\s+during|\s+for|$)',
            r'\bfrom\s+([\w\s]+?)\s+to\s+([\w\s]+?)(?:\s+in|\s+during|$)',
            r'\b(Q[1-4])[-–—](Q[1-4])\s+(\d{4})\b',  # Q1-Q3 2022
        ]
        
        for pattern in range_patterns:
            for match in re.finditer(pattern, query_text, re.IGNORECASE):
                if len(match.groups()) == 3:  # Q1-Q3 2022 format
                    start_q, end_q, year = match.groups()
                    filters.append({
                        'column': None,
                        'value': f"{start_q} {year}",
                        'operator': '>=',
                        'hint': 'time_period'
                    })
                    filters.append({
                        'column': None,
                        'value': f"{end_q} {year}",
                        'operator': '<=',
                        'hint': 'time_period'
                    })
                else:
                    # Regular range
                    filters.append({
                        'column': None,
                        'value': match.group(1).strip(),
                        'operator': '>=',
                        'hint': 'time_period'
                    })
                    filters.append({
                        'column': None,
                        'value': match.group(2).strip(),
                        'operator': '<=',
                        'hint': 'time_period'
                    })
        
        # Pattern 8: Comparative operators
        # Examples: "greater than 1000", "more than 50", "less than 100", "at least 500"
        comparison_patterns = [
            r'\b(?:greater|more)\s+than\s+(\d+(?:\.\d+)?)',
            r'\b(?:less|fewer)\s+than\s+(\d+(?:\.\d+)?)',
            r'\bat\s+least\s+(\d+(?:\.\d+)?)',
            r'\bat\s+most\s+(\d+(?:\.\d+)?)',
            r'\bexactly\s+(\d+(?:\.\d+)?)',
        ]
        
        for pattern in comparison_patterns:
            for match in re.finditer(pattern, query_text, re.IGNORECASE):
                value = match.group(1)
                if 'greater' in match.group(0).lower() or 'more' in match.group(0).lower():
                    operator = '>'
                elif 'less' in match.group(0).lower() or 'fewer' in match.group(0).lower():
                    operator = '<'
                elif 'at least' in match.group(0).lower():
                    operator = '>='
                elif 'at most' in match.group(0).lower():
                    operator = '<='
                else:
                    operator = '='
                
                filters.append({
                    'column': None,
                    'value': value,
                    'operator': operator,
                    'hint': 'numeric'
                })
        
        # Pattern 9: Special pharmaceutical terms and flags
        # Examples: "target prescribers", "PDRP enabled", "targeted accounts", "new growth decliner"
        pharma_patterns = [
            r'\b(target|targeted)\s+(?:prescribers?|accounts?|physicians?|doctors?)',
            r'\b(PDRP|pdrp)\s+(?:enabled|active|yes|flag)',
            r'\b(new|growth|decliner)\s+(?:prescribers?|accounts?)',
            r'\b(primary|secondary|tertiary)\s+(?:care|target)',
        ]
        
        for pattern in pharma_patterns:
            for match in re.finditer(pattern, query_text, re.IGNORECASE):
                flag_value = match.group(1)
                filters.append({
                    'column': None,
                    'value': flag_value,
                    'operator': '=',
                    'hint': 'flag'
                })
        
        # Pattern 10: Specialty and market patterns
        # Examples: "cardiology specialty", "retail market", "hospital accounts"
        specialty_patterns = [
            r'\b(cardiology|endocrinology|oncology|neurology|psychiatry|[\w]+ology)\s+(?:specialty|specialists?)',
            r'\b(retail|hospital|clinic|pharmacy)\s+(?:market|accounts?|channel)',
            r'\bspecialty\s+(?:is|=)?\s*["\']?(\w+)["\']?',
        ]
        
        for pattern in specialty_patterns:
            for match in re.finditer(pattern, query_text, re.IGNORECASE):
                specialty = match.group(1)
                filters.append({
                    'column': None,
                    'value': specialty,
                    'operator': '=',
                    'hint': 'specialty'
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
        hint_type = filter_hint.get('hint')  # 'product', 'time_period', etc.
        
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
        
        # If no column specified, search with hints
        else:
            # Use hint to narrow down column search
            if hint_type == 'product':
                # Search product-related columns (Product, ProductName, ProductGroup, etc.)
                target_columns = self._find_columns_by_hint(schema_context, ['product', 'item', 'drug', 'medication'])
            elif hint_type == 'time_period':
                # Search time-related columns (TimePeriod, Period, Date, Quarter, etc.)
                target_columns = self._find_columns_by_hint(schema_context, ['time', 'period', 'date', 'quarter', 'month', 'year'])
            else:
                # Search all text/enum columns
                target_columns = []
                for table_info in schema_context.get('tables', []):
                    table_name = table_info.get('name')
                    for column_info in table_info.get('columns', []):
                        col_name = column_info.get('name')
                        data_type = column_info.get('type', '').lower()
                        
                        # Only search text columns for name matches
                        if any(t in data_type for t in ['varchar', 'char', 'text', 'string']):
                            target_columns.append({
                                'table_name': table_name,
                                'column_name': col_name
                            })
            
            # Try to match value in target columns
            for col_info in target_columns:
                table_name = col_info['table_name']
                col_name = col_info['column_name']
                
                col_values = self._get_column_values(table_name, col_name)
                if col_values:
                    value_match = self._match_value(user_value, col_values)
                    if value_match and value_match.confidence > 0.7:  # Lower threshold for hinted searches
                        matches.append(value_match)
        
        return matches
    
    def _find_columns_by_hint(
        self,
        schema_context: Dict[str, Any],
        keywords: List[str]
    ) -> List[Dict[str, str]]:
        """Find columns whose names contain any of the keywords"""
        matching_columns = []
        
        for table_info in schema_context.get('tables', []):
            table_name = table_info.get('name')
            for column_info in table_info.get('columns', []):
                col_name = column_info.get('name', '')
                col_name_lower = col_name.lower()
                
                # Check if any keyword appears in column name
                if any(keyword in col_name_lower for keyword in keywords):
                    matching_columns.append({
                        'table_name': table_name,
                        'column_name': col_name
                    })
                    logger.info(f"  🎯 Found hint-matched column: {table_name}.{col_name}")
        
        return matching_columns
    
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
