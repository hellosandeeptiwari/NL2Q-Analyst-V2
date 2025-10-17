"""
Enhanced Schema Intelligence System with Deep Contextual Understanding
Provides intelligent analysis of database schema, relationships, and business context
"""

import os
import snowflake.connector
from typing import Dict, List, Any, Optional, Tuple, Set
import json
import re
from dataclasses import dataclass, field
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

@dataclass
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool
    table_name: str
    description: Optional[str] = None
    is_numeric: bool = False
    is_text: bool = False
    is_date: bool = False
    is_identifier: bool = False
    is_foreign_key: bool = False
    is_primary_key: bool = False
    is_amount_field: bool = False
    semantic_role: Optional[str] = None
    related_columns: List[str] = field(default_factory=list)
    sample_values: List[Any] = field(default_factory=list)
    business_meaning: Optional[str] = None
    data_patterns: Dict[str, Any] = field(default_factory=dict)
    # Enhanced temporal intelligence fields
    temporal_granularity: Optional[str] = None  # 'year', 'quarter', 'month', 'week', 'day', 'hour', 'minute', 'second'
    is_fiscal_period: bool = False  # True if fiscal period (FY, FQ) vs calendar period
    is_period_start: bool = False  # True if represents start of period
    is_period_end: bool = False  # True if represents end of period
    supports_time_series: bool = False  # True if suitable for time-series analysis
    temporal_context: Optional[str] = None  # Business context like 'reporting_period', 'transaction_date', 'effective_date'
    
    def __post_init__(self):
        self.is_numeric = self._is_numeric_type(self.data_type)
        self.is_text = self._is_text_type(self.data_type)
        self.is_date = self._is_date_type(self.data_type)
        self.is_identifier = self._is_identifier_column(self.name)
        self.is_amount_field = self._is_amount_field(self.name)
        self.semantic_role = self._determine_semantic_role()
        self.business_meaning = self._infer_business_meaning()
        # Enhanced temporal analysis
        if self.is_date:
            self._analyze_temporal_characteristics()
    
    def _is_numeric_type(self, data_type: str) -> bool:
        numeric_types = ['NUMBER', 'DECIMAL', 'NUMERIC', 'INT', 'INTEGER', 'BIGINT', 'SMALLINT', 'FLOAT', 'DOUBLE', 'REAL']
        return any(nt in data_type.upper() for nt in numeric_types)
    
    def _is_text_type(self, data_type: str) -> bool:
        text_types = ['VARCHAR', 'CHAR', 'TEXT', 'STRING']
        return any(tt in data_type.upper() for tt in text_types)
    
    def _is_date_type(self, data_type: str) -> bool:
        date_types = ['DATE', 'TIMESTAMP', 'TIME', 'DATETIME']
        return any(dt in data_type.upper() for dt in date_types)
    
    def _is_identifier_column(self, name: str) -> bool:
        """Detect if column is an identifier (ID, key, code)"""
        name_lower = name.lower()
        identifier_patterns = ['_id', 'id_', '_key', 'key_', '_code', 'code_', '_ref', 'ref_']
        return (name_lower.endswith('_id') or name_lower.startswith('id_') or 
                any(pattern in name_lower for pattern in identifier_patterns))
    
    def _is_amount_field(self, name: str) -> bool:
        """Detect if column represents monetary amounts"""
        name_lower = name.lower()
        amount_keywords = ['amount', 'rate', 'payment', 'price', 'cost', 'revenue', 'total', 'negotiated', 'fee', 'charge']
        return any(keyword in name_lower for keyword in amount_keywords) and self.is_numeric
    
    def _determine_semantic_role(self) -> str:
        """Determine the semantic role of this column"""
        name_lower = self.name.lower()
        
        if self.is_identifier:
            if 'provider' in name_lower:
                return 'provider_identifier'
            elif 'payer' in name_lower:
                return 'payer_identifier'
            elif 'service' in name_lower:
                return 'service_identifier'
            else:
                return 'general_identifier'
        
        if self.is_amount_field:
            if 'negotiated' in name_lower:
                return 'negotiated_payment'
            elif 'rate' in name_lower:
                return 'pricing_rate'
            elif 'revenue' in name_lower:
                return 'revenue_metric'
            else:
                return 'monetary_value'
        
        if 'name' in name_lower:
            return 'descriptive_name'
        elif 'description' in name_lower:
            return 'descriptive_text'
        elif 'file' in name_lower or 'source' in name_lower:
            return 'data_lineage'
        elif self.is_date:
            return 'temporal_marker'
        elif 'score' in name_lower or 'rating' in name_lower:
            return 'performance_metric'
        
        return 'general_attribute'
    
    def _infer_business_meaning(self) -> str:
        """Infer detailed business meaning from context"""
        table_lower = self.table_name.lower()
        col_lower = self.name.lower()
        
        # Healthcare pricing domain knowledge
        if 'provider' in table_lower:
            if 'provider_id' in col_lower:
                return "Unique identifier for healthcare providers - use for joining provider data across tables"
            elif 'payer' in col_lower and self.is_text:
                return "Insurance company name - use for grouping/filtering, NOT for mathematical calculations"
            elif 'tin' in col_lower:
                return "Tax Identification Number for provider organization"
        
        elif 'negotiated' in table_lower:
            if 'amount' in col_lower and self.is_numeric:
                return "Contracted payment amount between provider and payer - PRIMARY field for payment analysis"
            elif 'provider_id' in col_lower:
                return "Links to provider information - use for joining with PROVIDER_REFERENCES"
            elif 'service_code' in col_lower:
                return "Healthcare service identifier - use for joining with SERVICE_DEFINITIONS"
        
        elif 'rate' in table_lower:
            if 'amount' in col_lower and self.is_numeric:
                return "Standard payment rate - use for rate analysis and comparisons"
            elif 'rate_type' in col_lower:
                return "Classification of rate (standard, negotiated, etc.)"
        
        elif 'service' in table_lower:
            if 'service_code' in col_lower:
                return "Standardized healthcare service code - primary key for service lookup"
            elif 'description' in col_lower:
                return "Human-readable service description"
            elif 'base_rate' in col_lower and self.is_numeric:
                return "Standard pricing for this service before negotiations"
        
        elif 'volume' in table_lower:
            if 'volume_count' in col_lower and self.is_numeric:
                return "Number of times service was performed - use for utilization analysis"
            elif 'total_revenue' in col_lower and self.is_numeric:
                return "Aggregate revenue for service/provider combination"
        
        elif 'metric' in table_lower:
            if 'score' in col_lower and self.is_numeric:
                return "Performance measurement - use for quality analysis"
        
        # Generic meanings based on column patterns
        if self.is_amount_field:
            return f"Monetary value representing {col_lower.replace('_', ' ')} - use for financial calculations"
        elif self.is_identifier:
            return f"Unique identifier for {col_lower.replace('_id', '').replace('id_', '')} - use for table joins"
        elif self.is_text:
            return f"Text field containing {col_lower.replace('_', ' ')} - use for grouping/filtering only"
        elif self.is_numeric:
            return f"Numeric value for {col_lower.replace('_', ' ')} - can use in calculations"
        
        return f"Data field: {col_lower.replace('_', ' ')}"
    
    def _analyze_temporal_characteristics(self):
        """
        Enhanced temporal intelligence - analyze date/time column characteristics
        for better time-series analysis, trend detection, and temporal query planning
        """
        col_lower = self.name.lower()
        data_type_upper = self.data_type.upper()
        
        # Determine temporal granularity from data type and column name
        if 'TIMESTAMP' in data_type_upper or 'DATETIME' in data_type_upper:
            if any(term in col_lower for term in ['hour', 'minute', 'second', 'time']):
                self.temporal_granularity = 'hour'  # High-precision timestamps
            else:
                self.temporal_granularity = 'day'  # Standard timestamps
        elif 'DATE' in data_type_upper:
            self.temporal_granularity = 'day'  # Standard date columns
        elif 'TIME' in data_type_upper:
            self.temporal_granularity = 'second'  # Time-only columns
        
        # Refine granularity based on column naming patterns
        if any(term in col_lower for term in ['year', 'yyyy', 'yr', 'annual', 'annually']):
            self.temporal_granularity = 'year'
        elif any(term in col_lower for term in ['quarter', 'qtr', 'q1', 'q2', 'q3', 'q4', 'quarterly']):
            self.temporal_granularity = 'quarter'
        elif any(term in col_lower for term in ['month', 'mm', 'mon', 'monthly']):
            self.temporal_granularity = 'month'
        elif any(term in col_lower for term in ['week', 'wk', 'weekly']):
            self.temporal_granularity = 'week'
        elif any(term in col_lower for term in ['day', 'dd', 'daily', 'date']):
            self.temporal_granularity = 'day'
        
        # Detect fiscal vs calendar periods
        fiscal_indicators = ['fiscal', 'fy', 'fq', 'fm', 'fiscal_year', 'fiscal_quarter']
        self.is_fiscal_period = any(indicator in col_lower for indicator in fiscal_indicators)
        
        # Detect period start/end indicators
        self.is_period_start = any(term in col_lower for term in ['start', 'begin', 'from', 'effective', 'opening'])
        self.is_period_end = any(term in col_lower for term in ['end', 'close', 'closing', 'through', 'until', 'expiration'])
        
        # Determine if suitable for time-series analysis
        # Time-series appropriate for: transaction dates, reporting periods, event timestamps
        time_series_patterns = [
            'date', 'timestamp', 'created', 'updated', 'modified', 'recorded',
            'transaction', 'event', 'occurred', 'happened', 'reported',
            'period', 'month', 'quarter', 'year', 'day', 'week'
        ]
        self.supports_time_series = any(pattern in col_lower for pattern in time_series_patterns)
        
        # Determine temporal context for business understanding
        if any(term in col_lower for term in ['report', 'reporting', 'period']):
            self.temporal_context = 'reporting_period'
        elif any(term in col_lower for term in ['transaction', 'sale', 'purchase', 'payment']):
            self.temporal_context = 'transaction_date'
        elif any(term in col_lower for term in ['effective', 'valid', 'active']):
            self.temporal_context = 'effective_date'
        elif any(term in col_lower for term in ['created', 'insert', 'added']):
            self.temporal_context = 'creation_date'
        elif any(term in col_lower for term in ['modified', 'updated', 'changed']):
            self.temporal_context = 'modification_date'
        elif any(term in col_lower for term in ['expire', 'expiration', 'end', 'termination']):
            self.temporal_context = 'expiration_date'
        elif any(term in col_lower for term in ['start', 'begin', 'opening']):
            self.temporal_context = 'start_date'
        elif any(term in col_lower for term in ['due', 'deadline', 'target']):
            self.temporal_context = 'due_date'
        else:
            self.temporal_context = 'general_temporal'
        
        # Update business meaning with temporal insights
        if self.temporal_context and not self.business_meaning:
            granularity_text = self.temporal_granularity or 'date'
            period_type = 'fiscal' if self.is_fiscal_period else 'calendar'
            self.business_meaning = f"{period_type.capitalize()} {self.temporal_context.replace('_', ' ')} at {granularity_text} granularity"
            if self.supports_time_series:
                self.business_meaning += " - suitable for time-series analysis and trend detection"
    
    def get_temporal_query_hints(self) -> Dict[str, Any]:
        """
        Provide query hints for temporal operations based on column characteristics
        Used by query planner for intelligent temporal query generation
        """
        if not self.is_date:
            return {}
        
        hints = {
            'column_name': self.name,
            'granularity': self.temporal_granularity,
            'supports_aggregation': self.supports_time_series,
            'fiscal_period': self.is_fiscal_period,
            'temporal_context': self.temporal_context
        }
        
        # Suggest appropriate date functions based on granularity
        date_functions = []
        if self.temporal_granularity == 'year':
            date_functions = ['YEAR()', 'DATE_TRUNC(year)', 'EXTRACT(YEAR FROM)']
        elif self.temporal_granularity == 'quarter':
            date_functions = ['QUARTER()', 'DATE_TRUNC(quarter)', 'EXTRACT(QUARTER FROM)']
        elif self.temporal_granularity == 'month':
            date_functions = ['MONTH()', 'DATE_TRUNC(month)', 'EXTRACT(MONTH FROM)']
        elif self.temporal_granularity == 'week':
            date_functions = ['WEEK()', 'DATE_TRUNC(week)', 'WEEKOFYEAR()']
        elif self.temporal_granularity == 'day':
            date_functions = ['DATE()', 'DATE_TRUNC(day)', 'CAST(... AS DATE)']
        
        hints['recommended_functions'] = date_functions
        
        # Suggest window functions for time-series
        if self.supports_time_series:
            hints['window_functions'] = [
                f'ROW_NUMBER() OVER (ORDER BY {self.name})',
                f'LAG({self.name}) OVER (ORDER BY {self.name})',
                f'LEAD({self.name}) OVER (ORDER BY {self.name})',
                f'FIRST_VALUE(...) OVER (ORDER BY {self.name})',
                f'LAST_VALUE(...) OVER (ORDER BY {self.name})'
            ]
        
        # Suggest common temporal filters
        temporal_patterns = []
        if self.temporal_granularity in ['day', 'hour', 'minute', 'second']:
            temporal_patterns.extend([
                'last_7_days', 'last_30_days', 'last_90_days', 
                'current_month', 'current_quarter', 'current_year',
                'year_to_date', 'month_to_date'
            ])
        if self.temporal_granularity in ['month', 'quarter']:
            temporal_patterns.extend([
                'last_6_months', 'last_12_months', 'last_4_quarters',
                'current_fiscal_year' if self.is_fiscal_period else 'current_calendar_year'
            ])
        
        hints['common_filter_patterns'] = temporal_patterns
        
        return hints

@dataclass
class RelationshipInfo:
    from_table: str
    to_table: str
    from_column: str
    to_column: str
    relationship_type: str  # 'foreign_key', 'common_identifier', 'semantic_link'
    confidence: float  # 0.0 to 1.0
    description: str
    business_context: str

@dataclass
class TableInfo:
    name: str
    columns: List[ColumnInfo]
    description: Optional[str] = None
    business_purpose: Optional[str] = None
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[str] = field(default_factory=list)
    amount_columns: List[str] = field(default_factory=list)
    identifier_columns: List[str] = field(default_factory=list)
    relationships: List[RelationshipInfo] = field(default_factory=list)
    table_domain: Optional[str] = None
    data_volume_estimate: Optional[int] = None
    
    def __post_init__(self):
        self._classify_columns()
        self._infer_table_domain()
    
    def _classify_columns(self):
        """Classify columns by their roles"""
        self.primary_keys = []
        self.foreign_keys = []
        self.amount_columns = []
        self.identifier_columns = []
        
        for col in self.columns:
            if col.is_identifier:
                self.identifier_columns.append(col.name)
                # Infer primary vs foreign key
                if col.name.upper() == f"{self.name}_ID" or col.name.lower() in ['id', 'key']:
                    self.primary_keys.append(col.name)
                    col.is_primary_key = True
                else:
                    self.foreign_keys.append(col.name)
                    col.is_foreign_key = True
            
            if col.is_amount_field:
                self.amount_columns.append(col.name)
    
    def _infer_table_domain(self):
        """Infer what business domain this table belongs to"""
        name_lower = self.name.lower()
        
        domain_keywords = {
            'provider': ['provider', 'doctor', 'hospital', 'clinic'],
            'payer': ['payer', 'insurance', 'plan'],
            'service': ['service', 'procedure', 'treatment'],
            'financial': ['rate', 'payment', 'cost', 'revenue', 'negotiated'],
            'reference': ['reference', 'lookup', 'definition'],
            'metric': ['metric', 'score', 'performance', 'quality'],
            'volume': ['volume', 'utilization', 'count']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in name_lower for keyword in keywords):
                self.table_domain = domain
                break
        
        if not self.table_domain:
            self.table_domain = 'general'
    
    def get_numeric_columns(self) -> List[ColumnInfo]:
        return [col for col in self.columns if col.is_numeric]
    
    def get_text_columns(self) -> List[ColumnInfo]:
        return [col for col in self.columns if col.is_text]
    
    def get_key_columns(self) -> List[ColumnInfo]:
        """Find columns that look like IDs or keys"""
        return [col for col in self.columns if col.is_identifier]
    
    def find_amount_columns(self) -> List[ColumnInfo]:
        """Find columns that contain monetary amounts"""
        return [col for col in self.columns if col.is_amount_field]
    
    def get_joinable_columns(self) -> List[ColumnInfo]:
        """Get columns that can be used for joining with other tables"""
        return [col for col in self.columns if col.is_identifier or col.is_foreign_key]
    
    def get_aggregatable_columns(self) -> List[ColumnInfo]:
        """Get columns suitable for aggregation (SUM, AVG, etc.)"""
        return [col for col in self.columns if col.is_numeric and not col.is_identifier]
    
    def get_temporal_columns(self) -> List[ColumnInfo]:
        """Get all date/time columns with temporal intelligence"""
        return [col for col in self.columns if col.is_date]
    
    def get_time_series_columns(self) -> List[ColumnInfo]:
        """Get columns suitable for time-series analysis"""
        return [col for col in self.columns if col.is_date and col.supports_time_series]
    
    def get_primary_temporal_column(self) -> Optional[ColumnInfo]:
        """
        Identify the primary temporal column for time-based queries
        Priority: transaction dates > reporting periods > creation dates > any date
        """
        temporal_cols = self.get_temporal_columns()
        if not temporal_cols:
            return None
        
        # Priority scoring based on temporal context
        priority_contexts = {
            'transaction_date': 10,
            'reporting_period': 9,
            'effective_date': 8,
            'creation_date': 7,
            'modification_date': 6,
            'due_date': 5,
            'start_date': 4,
            'expiration_date': 3,
            'general_temporal': 1
        }
        
        # Score each temporal column
        scored_columns = []
        for col in temporal_cols:
            priority = priority_contexts.get(col.temporal_context or 'general_temporal', 1)
            # Boost if supports time-series
            if col.supports_time_series:
                priority += 2
            scored_columns.append((col, priority))
        
        # Return highest priority column
        scored_columns.sort(key=lambda x: x[1], reverse=True)
        return scored_columns[0][0]
    
    def supports_temporal_analysis(self) -> bool:
        """Check if table has sufficient temporal data for trend analysis"""
        return len(self.get_time_series_columns()) > 0
    
    def get_temporal_granularity_options(self) -> List[str]:
        """Get available temporal granularities for this table"""
        temporal_cols = self.get_temporal_columns()
        granularities = set()
        for col in temporal_cols:
            if col.temporal_granularity:
                granularities.add(col.temporal_granularity)
        return sorted(list(granularities))
    
    def has_fiscal_periods(self) -> bool:
        """Check if table contains fiscal period data"""
        return any(col.is_fiscal_period for col in self.columns if col.is_date)
    
    def get_temporal_query_suggestions(self) -> Dict[str, Any]:
        """
        Generate intelligent query suggestions for temporal analysis
        Used by query planner to suggest time-based query patterns
        """
        primary_temporal = self.get_primary_temporal_column()
        if not primary_temporal:
            return {}
        
        suggestions = {
            'primary_date_column': primary_temporal.name,
            'granularity': primary_temporal.temporal_granularity,
            'supports_trends': primary_temporal.supports_time_series,
            'fiscal_aware': primary_temporal.is_fiscal_period,
            'suggested_patterns': []
        }
        
        # Generate pattern suggestions based on granularity
        if primary_temporal.temporal_granularity in ['day', 'hour']:
            suggestions['suggested_patterns'].extend([
                f"Trends over time using {primary_temporal.name}",
                f"Month-over-month comparison by {primary_temporal.name}",
                f"Year-over-year comparison by {primary_temporal.name}",
                f"Rolling 30-day averages based on {primary_temporal.name}",
                f"Daily, weekly, or monthly aggregations by {primary_temporal.name}"
            ])
        elif primary_temporal.temporal_granularity in ['month', 'quarter']:
            suggestions['suggested_patterns'].extend([
                f"Quarterly trends using {primary_temporal.name}",
                f"Year-over-year quarterly comparison by {primary_temporal.name}",
                f"Seasonal patterns by {primary_temporal.name}",
                f"Annual aggregations by {primary_temporal.name}"
            ])
        elif primary_temporal.temporal_granularity == 'year':
            suggestions['suggested_patterns'].extend([
                f"Multi-year trends using {primary_temporal.name}",
                f"Year-over-year growth by {primary_temporal.name}",
                f"Historical analysis by {primary_temporal.name}"
            ])
        
        # Add fiscal-specific patterns
        if primary_temporal.is_fiscal_period:
            suggestions['suggested_patterns'].append(
                f"Fiscal year analysis using {primary_temporal.name}"
            )
        
        return suggestions
    
    def find_related_tables_by_column_name(self, all_tables: Dict[str, 'TableInfo']) -> List[Tuple[str, str, float]]:
        """Find tables that share column names (potential relationships)"""
        related = []
        my_columns = {col.name.upper() for col in self.columns}
        
        for other_table_name, other_table in all_tables.items():
            if other_table_name == self.name:
                continue
                
            other_columns = {col.name.upper() for col in other_table.columns}
            common_columns = my_columns.intersection(other_columns)
            
            if common_columns:
                # Calculate relationship confidence
                confidence = len(common_columns) / min(len(my_columns), len(other_columns))
                for common_col in common_columns:
                    related.append((other_table_name, common_col, confidence))
        
        return sorted(related, key=lambda x: x[2], reverse=True)
        return [col for col in self.columns if col.is_numeric]
    
    def get_text_columns(self) -> List[ColumnInfo]:
        return [col for col in self.columns if col.is_text]
    
    def get_key_columns(self) -> List[ColumnInfo]:
        """Find columns that look like IDs or keys"""
        return [col for col in self.columns if 'id' in col.name.lower()]
    
    def find_amount_columns(self) -> List[ColumnInfo]:
        """Find columns that contain monetary amounts"""
        amount_keywords = ['amount', 'rate', 'payment', 'price', 'cost', 'revenue', 'total', 'negotiated']
        return [col for col in self.columns if col.is_numeric and 
                any(keyword in col.name.lower() for keyword in amount_keywords)]

class EnhancedSchemaIntelligence:
    def __init__(self):
        self.tables: Dict[str, TableInfo] = {}
        self.relationships: List[RelationshipInfo] = []
        self.column_registry: Dict[str, List[Tuple[str, ColumnInfo]]] = defaultdict(list)  # column_name -> [(table, column_info)]
        self.semantic_patterns = self._build_semantic_patterns()
        
    def _build_semantic_patterns(self) -> Dict[str, Dict]:
        """Build patterns for understanding database semantics from names"""
        return {
            'healthcare_pricing': {
                'table_purposes': {
                    'provider_references': 'Master table of healthcare providers and their basic information',
                    'negotiated_rates': 'Contractual payment rates between providers and payers',
                    'all_rates': 'Standard or published rates for healthcare services',
                    'service_definitions': 'Master catalog of healthcare services and procedures',
                    'volume': 'Service utilization and volume metrics',
                    'metrics': 'Performance and quality measurements'
                },
                'relationship_patterns': {
                    'provider_id': {
                        'primary_table': 'provider_references',
                        'business_meaning': 'Healthcare provider unique identifier',
                        'typical_joins': ['negotiated_rates', 'all_rates', 'volume', 'metrics']
                    },
                    'service_code': {
                        'primary_table': 'service_definitions', 
                        'business_meaning': 'Healthcare service/procedure identifier',
                        'typical_joins': ['negotiated_rates', 'all_rates', 'volume']
                    },
                    'payer_id': {
                        'primary_table': None,
                        'business_meaning': 'Insurance payer identifier',
                        'typical_joins': ['negotiated_rates']
                    }
                },
                'amount_field_hierarchy': {
                    'negotiated_amount': {'priority': 1, 'meaning': 'Primary payment amount - contractual rate'},
                    'amount': {'priority': 2, 'meaning': 'Standard payment amount'},
                    'base_rate': {'priority': 3, 'meaning': 'Base service rate before adjustments'},
                    'total_revenue': {'priority': 4, 'meaning': 'Aggregate revenue amount'}
                },
                'business_rules': {
                    'payment_analysis': {
                        'primary_amount_source': 'negotiated_rates.negotiated_amount',
                        'fallback_amount_source': 'all_rates.amount',
                        'required_joins': ['provider_references -> negotiated_rates ON provider_id'],
                        'forbidden_operations': ['AVG on text fields', 'SUM on identifier fields']
                    }
                }
            }
        }
    
    async def discover_complete_schema(self) -> Dict[str, Any]:
        """Comprehensive schema discovery with intelligent relationship analysis"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            print("🧠 Starting Enhanced Schema Intelligence Discovery...")
            
            # 1. Discover all tables
            table_names = await self._discover_tables(cursor)
            print(f"📊 Found {len(table_names)} tables: {table_names}")
            
            # 2. Analyze each table deeply
            for table_name in table_names:
                print(f"🔍 Analyzing table: {table_name}")
                table_info = await self._analyze_table_deeply(cursor, table_name)
                self.tables[table_name] = table_info
                
                # Build column registry for relationship discovery
                for col in table_info.columns:
                    self.column_registry[col.name.upper()].append((table_name, col))
            
            # 3. Discover intelligent relationships
            print("🔗 Discovering table relationships...")
            self.relationships = await self._discover_intelligent_relationships()
            
            # 4. Generate contextual business intelligence
            print("💡 Generating business context...")
            business_context = self._generate_contextual_business_intelligence()
            
            # 5. Create enhanced query guidance
            print("📋 Creating query guidance...")
            query_guidance = self._generate_intelligent_query_guidance()
            
            cursor.close()
            conn.close()
            
            result = {
                'tables': {name: self._table_to_dict(table) for name, table in self.tables.items()},
                'relationships': [self._relationship_to_dict(rel) for rel in self.relationships],
                'business_context': business_context,
                'query_guidance': query_guidance,
                'column_analysis': self._generate_column_analysis(),
                'semantic_understanding': self._generate_semantic_understanding()
            }
            
            print("✅ Enhanced Schema Intelligence completed!")
            return result
            
        except Exception as e:
            print(f"❌ Enhanced schema discovery failed: {e}")
            return {}
    
    async def _discover_intelligent_relationships(self) -> List[RelationshipInfo]:
        """Discover relationships using multiple intelligence methods"""
        relationships = []
        
        print("🔍 Method 1: Column name matching analysis...")
        name_based_rels = self._discover_relationships_by_column_names()
        relationships.extend(name_based_rels)
        
        print("🔍 Method 2: Semantic pattern analysis...")
        semantic_rels = self._discover_relationships_by_semantic_patterns()
        relationships.extend(semantic_rels)
        
        print("🔍 Method 3: Business domain knowledge...")
        domain_rels = self._discover_relationships_by_domain_knowledge()
        relationships.extend(domain_rels)
        
        print(f"🔗 Discovered {len(relationships)} total relationships")
        return relationships
    
    def _discover_relationships_by_column_names(self) -> List[RelationshipInfo]:
        """Find relationships based on shared column names"""
        relationships = []
        
        for column_name, table_column_pairs in self.column_registry.items():
            if len(table_column_pairs) > 1:  # Column appears in multiple tables
                # Determine primary table (usually the one with simpler structure)
                primary_candidate = None
                foreign_candidates = []
                
                for table_name, col_info in table_column_pairs:
                    table_info = self.tables[table_name]
                    
                    # Primary table heuristics
                    if (col_info.name.upper() == f"{table_name}_ID" or 
                        'reference' in table_name.lower() or 
                        'definition' in table_name.lower() or
                        len(table_info.columns) < 10):  # Simpler table likely primary
                        primary_candidate = (table_name, col_info)
                    else:
                        foreign_candidates.append((table_name, col_info))
                
                # Create relationships
                if primary_candidate:
                    for foreign_table, foreign_col in foreign_candidates:
                        confidence = self._calculate_relationship_confidence(
                            primary_candidate[0], foreign_table, column_name)
                        
                        relationship = RelationshipInfo(
                            from_table=foreign_table,
                            to_table=primary_candidate[0],
                            from_column=column_name,
                            to_column=column_name,
                            relationship_type='foreign_key',
                            confidence=confidence,
                            description=f"{foreign_table}.{column_name} references {primary_candidate[0]}.{column_name}",
                            business_context=self._generate_relationship_business_context(
                                foreign_table, primary_candidate[0], column_name)
                        )
                        relationships.append(relationship)
        
        return relationships
    
    def _discover_relationships_by_semantic_patterns(self) -> List[RelationshipInfo]:
        """Find relationships using semantic understanding"""
        relationships = []
        patterns = self.semantic_patterns['healthcare_pricing']['relationship_patterns']
        
        for pattern_name, pattern_info in patterns.items():
            if pattern_name.upper() in self.column_registry:
                tables_with_column = [table for table, _ in self.column_registry[pattern_name.upper()]]
                primary_table = pattern_info.get('primary_table')
                
                if primary_table and primary_table.upper() in [t.upper() for t in tables_with_column]:
                    # Create relationships from primary to other tables
                    for table in tables_with_column:
                        if table.upper() != primary_table.upper():
                            relationship = RelationshipInfo(
                                from_table=table,
                                to_table=primary_table.upper(),
                                from_column=pattern_name.upper(),
                                to_column=pattern_name.upper(),
                                relationship_type='semantic_link',
                                confidence=0.9,
                                description=f"Semantic relationship: {table} references {primary_table} via {pattern_name}",
                                business_context=pattern_info['business_meaning']
                            )
                            relationships.append(relationship)
        
        return relationships
    
    def _discover_relationships_by_domain_knowledge(self) -> List[RelationshipInfo]:
        """Find relationships using healthcare domain knowledge"""
        relationships = []
        
        # Known healthcare pricing relationships
        known_relationships = [
            ('PROVIDER_REFERENCES', 'NEGOTIATED_RATES', 'PROVIDER_ID', 'Provider payment rates'),
            ('PROVIDER_REFERENCES', 'ALL_RATES', 'PROVIDER_ID', 'Provider standard rates'),
            ('PROVIDER_REFERENCES', 'VOLUME', 'PROVIDER_ID', 'Provider service volumes'),
            ('PROVIDER_REFERENCES', 'METRICS', 'PROVIDER_ID', 'Provider performance metrics'),
            ('SERVICE_DEFINITIONS', 'NEGOTIATED_RATES', 'SERVICE_CODE', 'Service rate definitions'),
            ('SERVICE_DEFINITIONS', 'ALL_RATES', 'SERVICE_CODE', 'Service standard rates'),
            ('SERVICE_DEFINITIONS', 'VOLUME', 'SERVICE_CODE', 'Service utilization')
        ]
        
        for primary_table, foreign_table, join_column, context in known_relationships:
            if (primary_table in self.tables and foreign_table in self.tables and
                join_column.upper() in self.column_registry):
                
                relationship = RelationshipInfo(
                    from_table=foreign_table,
                    to_table=primary_table,
                    from_column=join_column,
                    to_column=join_column,
                    relationship_type='domain_knowledge',
                    confidence=0.95,
                    description=f"Healthcare domain relationship: {foreign_table} -> {primary_table}",
                    business_context=context
                )
                relationships.append(relationship)
        
        return relationships
    
    def _calculate_relationship_confidence(self, primary_table: str, foreign_table: str, column_name: str) -> float:
        """Calculate confidence score for a relationship"""
        confidence = 0.5  # Base confidence
        
        # Column name patterns
        if column_name.endswith('_ID'):
            confidence += 0.3
        
        # Table name patterns
        if 'reference' in primary_table.lower() or 'definition' in primary_table.lower():
            confidence += 0.2
        
        # Table size heuristic (smaller table likely primary)
        primary_cols = len(self.tables[primary_table].columns)
        foreign_cols = len(self.tables[foreign_table].columns)
        if primary_cols < foreign_cols:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_relationship_business_context(self, from_table: str, to_table: str, column: str) -> str:
        """Generate business context for a relationship"""
        from_domain = self.tables[from_table].table_domain
        to_domain = self.tables[to_table].table_domain
        
        context_map = {
            ('financial', 'provider'): f"Financial data linked to provider via {column}",
            ('financial', 'service'): f"Financial data linked to service via {column}",
            ('metric', 'provider'): f"Performance metrics for provider via {column}",
            ('volume', 'provider'): f"Utilization data for provider via {column}",
            ('volume', 'service'): f"Service volume data via {column}"
        }
        
        return context_map.get((from_domain, to_domain), 
                              f"Data relationship between {from_table} and {to_table} via {column}")
    
    def _generate_contextual_business_intelligence(self) -> Dict[str, Any]:
        """Generate deep business intelligence about the schema"""
        return {
            'domain': 'healthcare_pricing_analytics',
            'entity_relationships': self._map_entity_relationships(),
            'data_flow_patterns': self._identify_data_flow_patterns(),
            'payment_analysis_guide': self._create_payment_analysis_guide(),
            'query_optimization_hints': self._generate_query_optimization_hints(),
            'business_rules': self._extract_business_rules()
        }
    
    def _map_entity_relationships(self) -> Dict[str, Any]:
        """Map how business entities relate to each other"""
        entity_map = {}
        
        for table_name, table_info in self.tables.items():
            entity_map[table_name] = {
                'domain': table_info.table_domain,
                'primary_purpose': table_info.business_purpose,
                'key_identifiers': table_info.identifier_columns,
                'amount_fields': table_info.amount_columns,
                'connects_to': [rel.to_table for rel in self.relationships if rel.from_table == table_name],
                'connected_from': [rel.from_table for rel in self.relationships if rel.to_table == table_name]
            }
        
        return entity_map
    
    def _create_payment_analysis_guide(self) -> Dict[str, Any]:
        """Create specific guidance for payment analysis queries"""
        amount_fields = []
        for table_name, table_info in self.tables.items():
            for col in table_info.find_amount_columns():
                amount_fields.append({
                    'table': table_name,
                    'column': col.name,
                    'business_meaning': col.business_meaning,
                    'semantic_role': col.semantic_role,
                    'priority': self._get_amount_field_priority(col.name)
                })
        
        # Sort by priority
        amount_fields.sort(key=lambda x: x['priority'])
        
        return {
            'primary_amount_sources': amount_fields,
            'recommended_joins_for_payment_analysis': [
                {
                    'pattern': 'provider_payment_analysis',
                    'tables': ['PROVIDER_REFERENCES', 'NEGOTIATED_RATES'],
                    'join_condition': 'PROVIDER_REFERENCES.PROVIDER_ID = NEGOTIATED_RATES.PROVIDER_ID',
                    'primary_amount_field': 'NEGOTIATED_RATES.NEGOTIATED_AMOUNT',
                    'use_case': 'Analyzing provider payment rates and comparing to averages'
                }
            ],
            'forbidden_operations': [
                'AVG(PAYER) - PAYER is text, not numeric',
                'SUM(PROVIDER_ID) - IDs are identifiers, not amounts',
                'Mathematical operations on VARCHAR fields'
            ]
        }
    
    def _get_amount_field_priority(self, column_name: str) -> int:
        """Get priority ranking for amount fields (1 = highest priority)"""
        priorities = {
            'negotiated_amount': 1,
            'amount': 2, 
            'base_rate': 3,
            'total_revenue': 4
        }
        
        col_lower = column_name.lower()
        for key, priority in priorities.items():
            if key in col_lower:
                return priority
        
        return 5  # Default low priority
    
    def _identify_data_flow_patterns(self) -> Dict[str, Any]:
        """Identify how data flows through the system"""
        return {
            'data_entry_points': ['PROVIDER_REFERENCES', 'SERVICE_DEFINITIONS'],
            'transaction_tables': ['NEGOTIATED_RATES', 'ALL_RATES'],
            'aggregation_tables': ['VOLUME', 'METRICS'],
            'typical_flow': 'Provider/Service -> Rates -> Volume/Metrics'
        }
    
    def _generate_query_optimization_hints(self) -> List[Dict]:
        """Generate query optimization hints"""
        return [
            {
                'pattern': 'payment_analysis',
                'hint': 'Use NEGOTIATED_RATES.NEGOTIATED_AMOUNT for primary payment data',
                'optimization': 'Index on PROVIDER_ID for fast joins'
            }
        ]
    
    def _extract_business_rules(self) -> List[Dict]:
        """Extract business rules from schema analysis"""
        return [
            {
                'rule': 'payment_calculation',
                'description': 'Use NEGOTIATED_AMOUNT for contracted rates',
                'enforcement': 'Always verify column data type before mathematical operations'
            }
        ]
    
    def _generate_intelligent_query_guidance(self) -> Dict[str, Any]:
        """Generate intelligent query guidance"""
        return {
            'data_type_rules': {
                'numeric_operations': 'Only use AVG/SUM on NUMBER/DECIMAL columns',
                'text_operations': 'Use COUNT/DISTINCT on VARCHAR columns',
                'forbidden': 'Never use AVG on text fields like PAYER'
            }
        }
    
    def _generate_column_analysis(self) -> Dict[str, Any]:
        """Generate column analysis"""
        return {
            'total_columns': sum(len(table.columns) for table in self.tables.values()),
            'amount_columns': sum(len(table.find_amount_columns()) for table in self.tables.values()),
            'identifier_columns': sum(len(table.get_key_columns()) for table in self.tables.values())
        }
    
    def _generate_semantic_understanding(self) -> Dict[str, Any]:
        """Generate semantic understanding"""
        return {
            'domain': 'healthcare_pricing',
            'confidence': 0.95,
            'key_insights': ['Provider-centric data model', 'Payment analysis capabilities']
        }
    
    def _relationship_to_dict(self, rel: RelationshipInfo) -> Dict:
        """Convert relationship to dict"""
        return {
            'from_table': rel.from_table,
            'to_table': rel.to_table,
            'from_column': rel.from_column,
            'to_column': rel.to_column,
            'relationship_type': rel.relationship_type,
            'confidence': rel.confidence,
            'description': rel.description,
            'business_context': rel.business_context
        }
    
    async def _discover_tables(self, cursor) -> List[str]:
        """Comprehensive schema discovery with deep understanding"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 1. Discover all tables
            table_names = await self._discover_tables(cursor)
            
            # 2. For each table, get detailed schema
            for table_name in table_names:
                table_info = await self._analyze_table_deeply(cursor, table_name)
                self.tables[table_name] = table_info
            
            # 3. Discover relationships
            self.relationships = await self._discover_relationships(cursor)
            
            # 4. Generate business context
            business_context = self._generate_business_context()
            
            cursor.close()
            conn.close()
            
            return {
                'tables': {name: self._table_to_dict(table) for name, table in self.tables.items()},
                'relationships': self.relationships,
                'business_context': business_context,
                'query_guidance': self._generate_query_guidance()
            }
            
        except Exception as e:
            print(f"❌ Enhanced schema discovery failed: {e}")
            return {}
    
    async def _discover_tables(self, cursor) -> List[str]:
        """Discover all available tables"""
        cursor.execute("SHOW TABLES")
        return [row[1] for row in cursor.fetchall()]
    
    async def _analyze_table_deeply(self, cursor, table_name: str) -> TableInfo:
        """Deep analysis of a single table"""
        # Get column metadata
        cursor.execute(f"DESCRIBE TABLE {table_name}")
        columns = []
        
        for row in cursor.fetchall():
            col_name, col_type, nullable = row[0], row[1], row[2] == 'Y'
            
            # Get sample values for better understanding
            sample_values = await self._get_sample_values(cursor, table_name, col_name)
            
            column_info = ColumnInfo(
                name=col_name,
                data_type=col_type,
                nullable=nullable,
                table_name=table_name,
                sample_values=sample_values,
                description=self._infer_column_description(col_name, col_type, sample_values)
            )
            columns.append(column_info)
        
        # Infer business purpose
        business_purpose = self._infer_business_purpose(table_name, columns)
        
        return TableInfo(
            name=table_name,
            columns=columns,
            business_purpose=business_purpose
        )
    
    async def _get_sample_values(self, cursor, table_name: str, col_name: str) -> List[Any]:
        """Get sample values to understand data patterns"""
        try:
            cursor.execute(f'SELECT DISTINCT "{col_name}" FROM "{table_name}" LIMIT 3')
            return [row[0] for row in cursor.fetchall()]
        except:
            return []
    
    def _infer_column_description(self, col_name: str, col_type: str, sample_values: List) -> str:
        """Infer what a column represents based on name, type, and samples"""
        name_lower = col_name.lower()
        
        # ID columns
        if 'id' in name_lower:
            return f"Unique identifier for {name_lower.replace('_id', '').replace('id', '')}"
        
        # Amount/Payment columns
        if any(word in name_lower for word in ['amount', 'rate', 'payment', 'cost', 'price']):
            return f"Monetary value representing {name_lower.replace('_', ' ')}"
        
        # Name/Description columns
        if 'name' in name_lower or 'description' in name_lower:
            return f"Text description or name field"
        
        # Code columns
        if 'code' in name_lower:
            return f"Standardized code for {name_lower.replace('_code', '').replace('code', '')}"
        
        # File/Source columns
        if any(word in name_lower for word in ['file', 'source']):
            return f"Data source or file reference"
        
        # Default based on type
        if 'VARCHAR' in col_type:
            return f"Text field containing {name_lower.replace('_', ' ')}"
        elif any(nt in col_type for nt in ['NUMBER', 'DECIMAL', 'INT']):
            return f"Numeric value for {name_lower.replace('_', ' ')}"
        
        return f"Data field: {name_lower.replace('_', ' ')}"
    
    def _infer_business_purpose(self, table_name: str, columns: List[ColumnInfo]) -> str:
        """Infer what business purpose this table serves"""
        name_lower = table_name.lower()
        
        if 'provider' in name_lower and 'reference' in name_lower:
            return "Healthcare provider identification and basic information"
        elif 'negotiated' in name_lower and 'rate' in name_lower:
            return "Contractual payment rates negotiated between providers and payers"
        elif 'rate' in name_lower:
            return "Payment rates and pricing information"
        elif 'service' in name_lower:
            return "Healthcare service definitions and classifications"
        elif 'volume' in name_lower:
            return "Service volume and utilization metrics"
        elif 'metric' in name_lower:
            return "Performance and quality metrics"
        
        return f"Data table for {name_lower.replace('_', ' ')}"
    
    async def _discover_relationships(self, cursor) -> List[Dict]:
        """Discover relationships between tables"""
        relationships = []
        
        # Use business domain knowledge
        if 'healthcare_pricing' in self.business_domain_knowledge:
            common_joins = self.business_domain_knowledge['healthcare_pricing']['common_joins']
            
            for table1, table2, join_column in common_joins:
                if table1.upper() in self.tables and table2.upper() in self.tables:
                    relationships.append({
                        'table1': table1.upper(),
                        'table2': table2.upper(),
                        'join_column': join_column.upper(),
                        'relationship_type': 'foreign_key',
                        'description': f"Join {table1} and {table2} on {join_column}"
                    })
        
        return relationships
    
    def _generate_business_context(self) -> Dict[str, Any]:
        """Generate business context for better query understanding"""
        return {
            'domain': 'healthcare_pricing',
            'key_concepts': {
                'providers': 'Healthcare organizations that deliver services',
                'payers': 'Insurance companies that pay for services',
                'negotiated_rates': 'Contracted payment amounts between providers and payers',
                'service_codes': 'Standardized codes for healthcare services'
            },
            'payment_analysis_guidance': {
                'payment_amount_sources': [
                    'NEGOTIATED_RATES.NEGOTIATED_AMOUNT (primary)',
                    'ALL_RATES.AMOUNT (alternative)',
                    'VOLUME.TOTAL_REVENUE (aggregate)'
                ],
                'provider_identification': 'PROVIDER_REFERENCES.PROVIDER_ID',
                'typical_joins': [
                    'PROVIDER_REFERENCES ← NEGOTIATED_RATES (on PROVIDER_ID)',
                    'SERVICE_DEFINITIONS ← NEGOTIATED_RATES (on SERVICE_CODE)'
                ]
            }
        }
    
    def _generate_query_guidance(self) -> Dict[str, Any]:
        """Generate specific guidance for query construction"""
        guidance = {
            'data_type_rules': {
                'numeric_aggregations': {
                    'allowed_functions': ['AVG', 'SUM', 'COUNT', 'MIN', 'MAX'],
                    'numeric_columns': []
                },
                'text_operations': {
                    'allowed_functions': ['COUNT', 'DISTINCT'],
                    'text_columns': []
                }
            },
            'join_recommendations': [],
            'business_logic_patterns': {}
        }
        
        # Populate with actual table data
        for table_name, table_info in self.tables.items():
            # Add numeric columns
            numeric_cols = [col.name for col in table_info.get_numeric_columns()]
            guidance['data_type_rules']['numeric_aggregations']['numeric_columns'].extend(
                [f"{table_name}.{col}" for col in numeric_cols]
            )
            
            # Add text columns
            text_cols = [col.name for col in table_info.get_text_columns()]
            guidance['data_type_rules']['text_operations']['text_columns'].extend(
                [f"{table_name}.{col}" for col in text_cols]
            )
        
        # Add join recommendations
        guidance['join_recommendations'] = [
            {
                'pattern': 'provider_payment_analysis',
                'tables': ['PROVIDER_REFERENCES', 'NEGOTIATED_RATES'],
                'join_condition': 'PROVIDER_REFERENCES.PROVIDER_ID = NEGOTIATED_RATES.PROVIDER_ID',
                'use_case': 'When analyzing provider payments or rates'
            }
        ]
        
        return guidance
    
    def _table_to_dict(self, table: TableInfo) -> Dict:
        """Convert TableInfo to dictionary for JSON serialization"""
        return {
            'name': table.name,
            'business_purpose': table.business_purpose,
            'columns': [
                {
                    'name': col.name,
                    'data_type': col.data_type,
                    'is_numeric': col.is_numeric,
                    'is_text': col.is_text,
                    'description': col.description,
                    'sample_values': col.sample_values[:3] if col.sample_values else []
                }
                for col in table.columns
            ],
            'numeric_columns': [col.name for col in table.get_numeric_columns()],
            'amount_columns': [col.name for col in table.find_amount_columns()],
            'key_columns': [col.name for col in table.get_key_columns()]
        }
    
    def _get_connection(self):
        """Get Snowflake connection"""
        return snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DATABASE'),
            schema=os.getenv('SNOWFLAKE_SCHEMA')
        )

    def generate_enhanced_system_prompt(self, schema_data: Dict) -> str:
        """Generate enhanced system prompt with deep schema understanding"""
        db_name = os.getenv("SNOWFLAKE_DATABASE", "HEALTHCARE_PRICING_ANALYTICS_SAMPLE")
        schema_name = os.getenv("SNOWFLAKE_SCHEMA", "SAMPLES")
        
        # Build detailed schema context
        schema_details = []
        for table_name, table_info in schema_data['tables'].items():
            schema_details.append(f"""
TABLE: {table_name}
PURPOSE: {table_info['business_purpose']}
COLUMNS:
{self._format_columns_for_prompt(table_info['columns'])}
NUMERIC COLUMNS (can use AVG/SUM): {', '.join(table_info['numeric_columns'])}
AMOUNT COLUMNS (payment/rate data): {', '.join(table_info['amount_columns'])}
KEY COLUMNS (for joining): {', '.join(table_info['key_columns'])}""")
        
        # Build relationship context
        relationship_context = "\n".join([
            f"- {rel['table1']} ← {rel['table2']} ON {rel['join_column']}: {rel['description']}"
            for rel in schema_data['relationships']
        ])
        
        # Build business context
        business_context = schema_data['business_context']
        payment_guidance = business_context['payment_analysis_guidance']
        
        system_prompt = f"""You are an expert SQL generator with deep understanding of healthcare pricing data.

DATABASE CONTEXT:
- Engine: Snowflake
- Database: {db_name}
- Schema: {schema_name}
- Domain: {business_context['domain']}

DETAILED SCHEMA WITH DATA TYPES:
{chr(10).join(schema_details)}

TABLE RELATIONSHIPS:
{relationship_context}

BUSINESS DOMAIN KNOWLEDGE:
{self._format_business_knowledge(business_context['key_concepts'])}

CRITICAL DATA TYPE RULES:
1. ONLY use AVG/SUM/mathematical operations on NUMERIC columns
2. TEXT columns (VARCHAR) can only use COUNT, DISTINCT, GROUP BY
3. For payment analysis, use these NUMERIC amount columns:
   {', '.join(payment_guidance['payment_amount_sources'])}

JOIN PATTERNS FOR COMMON QUERIES:
- Provider payments: {payment_guidance['typical_joins'][0]}
- Service rates: {payment_guidance['typical_joins'][1]}

QUERY CONSTRUCTION RULES:
1. Always verify column data types before using in calculations
2. Use proper table qualification: "{db_name}"."{schema_name}"."TABLE_NAME"
3. Join tables when you need data from multiple sources
4. Use meaningful aliases for readability
5. Add LIMIT for safety

When generating SQL:
- Understand the business question first
- Identify which tables contain the needed data
- Verify data types match your operations
- Use appropriate joins based on relationships
- Return only executable SQL"""

        return system_prompt
    
    def _format_columns_for_prompt(self, columns: List[Dict]) -> str:
        """Format columns with types and descriptions for LLM prompt"""
        formatted = []
        for col in columns:
            type_info = f"({col['data_type']})"
            numeric_flag = " [NUMERIC]" if col['is_numeric'] else ""
            text_flag = " [TEXT]" if col['is_text'] else ""
            sample_info = f" Examples: {col['sample_values']}" if col['sample_values'] else ""
            
            formatted.append(f"  - {col['name']} {type_info}{numeric_flag}{text_flag}: {col['description']}{sample_info}")
        
        return "\n".join(formatted)
    
    def _format_business_knowledge(self, concepts: Dict) -> str:
        """Format business knowledge for the prompt"""
        return "\n".join([f"- {key}: {value}" for key, value in concepts.items()])

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_enhanced_schema():
        intelligence = EnhancedSchemaIntelligence()
        schema_data = await intelligence.discover_complete_schema()
        
        print("🧠 ENHANCED SCHEMA INTELLIGENCE RESULTS:")
        print("=" * 80)
        
        # Save detailed schema
        with open('enhanced_schema_intelligence.json', 'w') as f:
            json.dump(schema_data, f, indent=2, default=str)
        
        print(f"📊 Discovered {len(schema_data['tables'])} tables with deep analysis")
        print(f"🔗 Found {len(schema_data['relationships'])} relationships")
        
        # Generate enhanced prompt
        enhanced_prompt = intelligence.generate_enhanced_system_prompt(schema_data)
        
        print("\n🤖 ENHANCED SYSTEM PROMPT:")
        print("-" * 50)
        print(enhanced_prompt)
        
        return schema_data, enhanced_prompt
    
    # Run the test
    asyncio.run(test_enhanced_schema())
