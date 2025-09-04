"""
Auto-Discovery Schema Tool with Business Intelligence
Discovers tables, columns, joins, enums, date grains, and metrics automatically
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import snowflake.connector
from collections import defaultdict

@dataclass
class TableInfo:
    """Enhanced table information with relationships"""
    table_name: str
    schema_name: str
    database_name: str
    table_type: str  # TABLE, VIEW, MATERIALIZED_VIEW
    row_count: Optional[int]
    size_bytes: Optional[int]
    last_updated: Optional[datetime]
    description: Optional[str]
    business_purpose: Optional[str]  # Auto-inferred
    data_freshness: str  # REAL_TIME, DAILY, WEEKLY, etc.
    primary_keys: List[str]
    foreign_keys: List[Dict[str, str]]
    related_tables: List[str]

@dataclass  
class ColumnInfo:
    """Enhanced column information with business metadata"""
    column_name: str
    data_type: str
    is_nullable: bool
    default_value: Optional[str]
    description: Optional[str]
    business_name: Optional[str]  # "Patient ID" vs "PATIENT_ID" 
    column_purpose: str  # ID, DIMENSION, MEASURE, DATE, FLAG
    cardinality: Optional[int]  # Distinct value count
    sample_values: List[str]
    value_ranges: Optional[Dict[str, Any]]  # min/max for numeric
    enum_values: Optional[List[str]]  # For categorical
    date_grain: Optional[str]  # DAY, WEEK, MONTH, QUARTER, YEAR
    business_synonyms: List[str]  # Alternative names
    pii_classification: str  # NONE, LOW, MEDIUM, HIGH
    
@dataclass
class RelationshipInfo:
    """Table relationship with join conditions"""
    from_table: str
    to_table: str
    join_type: str  # INNER, LEFT, RIGHT, FULL
    join_condition: str
    relationship_type: str  # ONE_TO_ONE, ONE_TO_MANY, MANY_TO_MANY
    confidence_score: float  # 0.0 to 1.0

@dataclass
class BusinessMetric:
    """Auto-discovered business metrics"""
    metric_name: str
    metric_type: str  # COUNT, SUM, AVG, RATE, RATIO
    base_table: str
    calculation_logic: str
    grain: str  # Patient-level, HCP-level, Product-level
    typical_filters: List[str]
    business_definition: str
    
class AutoDiscoverySchema:
    """
    Intelligent schema discovery for pharmaceutical analytics
    """
    
    def __init__(self, connection_params: Dict[str, str]):
        self.connection_params = connection_params
        self.discovered_tables: Dict[str, TableInfo] = {}
        self.discovered_columns: Dict[str, List[ColumnInfo]] = {}
        self.discovered_relationships: List[RelationshipInfo] = []
        self.discovered_metrics: List[BusinessMetric] = []
        
        # Pharmaceutical domain patterns
        self.pharma_patterns = {
            "patient_tables": [
                r".*patient.*", r".*member.*", r".*enrollee.*", 
                r".*subject.*", r".*person.*"
            ],
            "prescription_tables": [
                r".*rx.*", r".*script.*", r".*prescription.*", 
                r".*claim.*", r".*transaction.*"
            ],
            "hcp_tables": [
                r".*hcp.*", r".*physician.*", r".*prescriber.*", 
                r".*provider.*", r".*doctor.*"
            ],
            "product_tables": [
                r".*product.*", r".*drug.*", r".*ndc.*", 
                r".*therapeutic.*", r".*brand.*"
            ],
            "date_dimensions": [
                r".*date.*", r".*time.*", r".*calendar.*", 
                r".*period.*"
            ]
        }
        
        # Common pharma metrics
        self.pharma_metrics = {
            "NBRx": {
                "calculation": "COUNT(DISTINCT prescription_id WHERE first_fill = 'Y')",
                "definition": "New Brand Prescriptions - First time prescriptions for a brand"
            },
            "TRx": {
                "calculation": "COUNT(DISTINCT prescription_id)",
                "definition": "Total Prescriptions including refills"
            },
            "Writers": {
                "calculation": "COUNT(DISTINCT prescriber_id WHERE rx_count > 0)",
                "definition": "Number of unique prescribers who wrote prescriptions"
            },
            "Market_Share": {
                "calculation": "SUM(brand_rx) / SUM(total_class_rx) * 100",
                "definition": "Brand's share of total therapeutic class prescriptions"
            },
            "Persistence": {
                "calculation": "COUNT(patients_with_90day_coverage) / COUNT(total_new_patients)",
                "definition": "Percentage of new patients still on therapy after 90 days"
            }
        }
    
    async def discover_complete_schema(
        self, 
        target_schemas: List[str] = None,
        include_samples: bool = True,
        discover_relationships: bool = True
    ) -> Dict[str, Any]:
        """
        Complete schema discovery with business intelligence
        """
        
        discovery_results = {
            "tables": {},
            "columns": {},
            "relationships": [],
            "metrics": [],
            "business_glossary": {},
            "discovery_metadata": {
                "timestamp": datetime.now().isoformat(),
                "schemas_analyzed": target_schemas or ["ALL"],
                "discovery_duration_seconds": 0
            }
        }
        
        start_time = datetime.now()
        
        try:
            # Step 1: Discover all tables
            tables = await self._discover_tables(target_schemas)
            
            # Step 2: Analyze each table in detail
            for table_name, table_info in tables.items():
                # Get column information
                columns = await self._discover_columns(
                    table_info.database_name,
                    table_info.schema_name, 
                    table_info.table_name,
                    include_samples=include_samples
                )
                
                discovery_results["tables"][table_name] = asdict(table_info)
                discovery_results["columns"][table_name] = [asdict(col) for col in columns]
            
            # Step 3: Discover relationships
            if discover_relationships:
                relationships = await self._discover_relationships(tables)
                discovery_results["relationships"] = [asdict(rel) for rel in relationships]
            
            # Step 4: Auto-discover business metrics
            metrics = await self._discover_business_metrics(tables, discovery_results["columns"])
            discovery_results["metrics"] = [asdict(metric) for metric in metrics]
            
            # Step 5: Build business glossary
            glossary = await self._build_business_glossary(
                discovery_results["columns"], 
                discovery_results["metrics"]
            )
            discovery_results["business_glossary"] = glossary
            
            # Update metadata
            end_time = datetime.now()
            discovery_results["discovery_metadata"]["discovery_duration_seconds"] = (
                end_time - start_time
            ).total_seconds()
            discovery_results["discovery_metadata"]["tables_discovered"] = len(tables)
            discovery_results["discovery_metadata"]["relationships_found"] = len(relationships)
            discovery_results["discovery_metadata"]["metrics_identified"] = len(metrics)
            
            return discovery_results
            
        except Exception as e:
            raise Exception(f"Schema discovery failed: {str(e)}")
    
    async def _discover_tables(self, target_schemas: List[str] = None) -> Dict[str, TableInfo]:
        """Discover all relevant tables with metadata"""
        
        tables = {}
        
        # Base query for table discovery
        base_query = """
        SELECT 
            t.table_catalog as database_name,
            t.table_schema as schema_name,
            t.table_name,
            t.table_type,
            ts.row_count,
            ts.bytes as size_bytes,
            ts.last_altered as last_updated,
            c.comment as description
        FROM information_schema.tables t
        LEFT JOIN information_schema.table_storage_metrics ts 
            ON t.table_catalog = ts.table_catalog 
            AND t.table_schema = ts.table_schema 
            AND t.table_name = ts.table_name
        LEFT JOIN information_schema.tables c 
            ON t.table_catalog = c.table_catalog 
            AND t.table_schema = c.table_schema 
            AND t.table_name = c.table_name
        WHERE t.table_type IN ('BASE TABLE', 'VIEW')
        """
        
        # Add schema filter if specified
        if target_schemas:
            schema_list = "', '".join(target_schemas)
            base_query += f" AND t.table_schema IN ('{schema_list}')"
        
        base_query += " ORDER BY ts.row_count DESC NULLS LAST"
        
        # Execute query
        conn = snowflake.connector.connect(**self.connection_params)
        cursor = conn.cursor()
        
        try:
            cursor.execute(base_query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            for row in results:
                row_dict = dict(zip(columns, row))
                
                table_key = f"{row_dict['DATABASE_NAME']}.{row_dict['SCHEMA_NAME']}.{row_dict['TABLE_NAME']}"
                
                # Classify table purpose
                business_purpose = self._classify_table_purpose(row_dict['TABLE_NAME'])
                
                # Determine data freshness (simplified)
                data_freshness = self._determine_data_freshness(
                    row_dict['TABLE_NAME'], 
                    row_dict.get('LAST_UPDATED')
                )
                
                tables[table_key] = TableInfo(
                    table_name=row_dict['TABLE_NAME'],
                    schema_name=row_dict['SCHEMA_NAME'], 
                    database_name=row_dict['DATABASE_NAME'],
                    table_type=row_dict['TABLE_TYPE'],
                    row_count=row_dict.get('ROW_COUNT'),
                    size_bytes=row_dict.get('SIZE_BYTES'),
                    last_updated=row_dict.get('LAST_UPDATED'),
                    description=row_dict.get('DESCRIPTION'),
                    business_purpose=business_purpose,
                    data_freshness=data_freshness,
                    primary_keys=[],  # Will be populated separately
                    foreign_keys=[],  # Will be populated separately  
                    related_tables=[]  # Will be populated from relationships
                )
        
        finally:
            cursor.close()
            conn.close()
        
        return tables
    
    async def _discover_columns(
        self, 
        database: str, 
        schema: str, 
        table: str,
        include_samples: bool = True
    ) -> List[ColumnInfo]:
        """Discover column metadata with business intelligence"""
        
        columns = []
        
        # Get basic column information
        column_query = f"""
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default,
            comment as description,
            ordinal_position
        FROM information_schema.columns 
        WHERE table_catalog = '{database}' 
            AND table_schema = '{schema}' 
            AND table_name = '{table}'
        ORDER BY ordinal_position
        """
        
        conn = snowflake.connector.connect(**self.connection_params)
        cursor = conn.cursor()
        
        try:
            cursor.execute(column_query)
            results = cursor.fetchall()
            column_info = [desc[0] for desc in cursor.description]
            
            for row in results:
                row_dict = dict(zip(column_info, row))
                column_name = row_dict['COLUMN_NAME']
                
                # Classify column purpose
                column_purpose = self._classify_column_purpose(
                    column_name, row_dict['DATA_TYPE']
                )
                
                # Generate business name
                business_name = self._generate_business_name(column_name)
                
                # Get sample values and statistics if requested
                sample_values = []
                cardinality = None
                value_ranges = None
                enum_values = None
                
                if include_samples:
                    stats = await self._get_column_statistics(
                        database, schema, table, column_name, row_dict['DATA_TYPE']
                    )
                    sample_values = stats.get('sample_values', [])
                    cardinality = stats.get('cardinality')
                    value_ranges = stats.get('value_ranges')
                    enum_values = stats.get('enum_values')
                
                # Determine date grain for date columns
                date_grain = self._determine_date_grain(column_name, row_dict['DATA_TYPE'])
                
                # Generate business synonyms
                business_synonyms = self._generate_business_synonyms(column_name)
                
                # Classify PII risk
                pii_classification = self._classify_pii_risk(column_name)
                
                columns.append(ColumnInfo(
                    column_name=column_name,
                    data_type=row_dict['DATA_TYPE'],
                    is_nullable=row_dict['IS_NULLABLE'] == 'YES',
                    default_value=row_dict['COLUMN_DEFAULT'],
                    description=row_dict.get('DESCRIPTION'),
                    business_name=business_name,
                    column_purpose=column_purpose,
                    cardinality=cardinality,
                    sample_values=sample_values,
                    value_ranges=value_ranges,
                    enum_values=enum_values,
                    date_grain=date_grain,
                    business_synonyms=business_synonyms,
                    pii_classification=pii_classification
                ))
        
        finally:
            cursor.close()
            conn.close()
        
        return columns
    
    async def _get_column_statistics(
        self, 
        database: str, 
        schema: str, 
        table: str, 
        column: str,
        data_type: str
    ) -> Dict[str, Any]:
        """Get detailed column statistics and sample values"""
        
        stats = {}
        full_table_name = f"{database}.{schema}.{table}"
        
        conn = snowflake.connector.connect(**self.connection_params)
        cursor = conn.cursor()
        
        try:
            # Basic statistics query
            if 'NUMBER' in data_type or 'FLOAT' in data_type or 'DECIMAL' in data_type:
                # Numeric statistics
                stats_query = f"""
                SELECT 
                    COUNT(DISTINCT "{column}") as cardinality,
                    MIN("{column}") as min_value,
                    MAX("{column}") as max_value,
                    AVG("{column}") as avg_value
                FROM {full_table_name}
                WHERE "{column}" IS NOT NULL
                """
                
                cursor.execute(stats_query)
                result = cursor.fetchone()
                
                if result:
                    stats['cardinality'] = result[0]
                    stats['value_ranges'] = {
                        'min': result[1],
                        'max': result[2], 
                        'avg': result[3]
                    }
            
            else:
                # Text/categorical statistics
                stats_query = f"""
                SELECT 
                    COUNT(DISTINCT "{column}") as cardinality
                FROM {full_table_name}
                WHERE "{column}" IS NOT NULL
                """
                
                cursor.execute(stats_query)
                result = cursor.fetchone()
                
                if result:
                    stats['cardinality'] = result[0]
                    
                    # Get enum values if low cardinality
                    if result[0] and result[0] <= 50:
                        enum_query = f"""
                        SELECT DISTINCT "{column}" 
                        FROM {full_table_name}
                        WHERE "{column}" IS NOT NULL
                        ORDER BY "{column}"
                        LIMIT 50
                        """
                        
                        cursor.execute(enum_query)
                        enum_results = cursor.fetchall()
                        stats['enum_values'] = [row[0] for row in enum_results]
            
            # Sample values (always get some)
            sample_query = f"""
            SELECT DISTINCT "{column}"
            FROM {full_table_name}
            WHERE "{column}" IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 10
            """
            
            cursor.execute(sample_query)
            sample_results = cursor.fetchall()
            stats['sample_values'] = [str(row[0]) for row in sample_results]
        
        except Exception as e:
            # If statistics fail, return empty stats
            stats = {'sample_values': [], 'cardinality': None}
        
        finally:
            cursor.close()
            conn.close()
        
        return stats
    
    def _classify_table_purpose(self, table_name: str) -> str:
        """Classify table business purpose based on name patterns"""
        
        table_lower = table_name.lower()
        
        for purpose, patterns in self.pharma_patterns.items():
            for pattern in patterns:
                if re.match(pattern, table_lower):
                    return purpose.replace('_', ' ').title()
        
        # General classification
        if any(word in table_lower for word in ['fact', 'transaction', 'activity']):
            return "Fact Table"
        elif any(word in table_lower for word in ['dim', 'dimension', 'lookup', 'master']):
            return "Dimension Table"
        elif any(word in table_lower for word in ['agg', 'summary', 'rollup']):
            return "Aggregate Table"
        else:
            return "Reference Table"
    
    def _classify_column_purpose(self, column_name: str, data_type: str) -> str:
        """Classify column purpose for analytics"""
        
        column_lower = column_name.lower()
        
        # ID columns
        if column_lower.endswith('_id') or column_lower.endswith('id') or 'key' in column_lower:
            return "ID"
        
        # Date columns
        if 'date' in data_type.lower() or 'timestamp' in data_type.lower():
            return "DATE"
        
        # Measure columns (numeric that can be aggregated)
        if ('NUMBER' in data_type or 'DECIMAL' in data_type or 'FLOAT' in data_type):
            if any(word in column_lower for word in ['count', 'amount', 'total', 'sum', 'qty', 'quantity', 'volume']):
                return "MEASURE"
            else:
                return "NUMERIC_DIMENSION"
        
        # Flag columns
        if any(word in column_lower for word in ['flag', 'ind', 'indicator', 'is_', 'has_']):
            return "FLAG"
        
        # Default to dimension
        return "DIMENSION"
    
    def _generate_business_name(self, column_name: str) -> str:
        """Convert technical column name to business-friendly name"""
        
        # Remove prefixes/suffixes
        clean_name = re.sub(r'^(dim_|fact_|stg_)', '', column_name.lower())
        clean_name = re.sub(r'(_id|_key|_cd|_desc)$', '', clean_name)
        
        # Split on underscores and capitalize
        words = clean_name.split('_')
        business_words = []
        
        for word in words:
            # Handle common abbreviations
            word_mappings = {
                'hcp': 'Healthcare Provider',
                'ndc': 'NDC',
                'rx': 'Prescription', 
                'qty': 'Quantity',
                'amt': 'Amount',
                'nbr': 'Number',
                'pct': 'Percent',
                'avg': 'Average',
                'max': 'Maximum',
                'min': 'Minimum'
            }
            
            business_words.append(word_mappings.get(word, word.capitalize()))
        
        return ' '.join(business_words)
    
    def _determine_date_grain(self, column_name: str, data_type: str) -> Optional[str]:
        """Determine date granularity from column name and type"""
        
        if 'date' not in data_type.lower() and 'timestamp' not in data_type.lower():
            return None
        
        column_lower = column_name.lower()
        
        if any(word in column_lower for word in ['year', 'yr']):
            return "YEAR"
        elif any(word in column_lower for word in ['quarter', 'qtr']):
            return "QUARTER"  
        elif any(word in column_lower for word in ['month', 'mon']):
            return "MONTH"
        elif any(word in column_lower for word in ['week', 'wk']):
            return "WEEK"
        elif 'timestamp' in data_type.lower():
            return "TIMESTAMP"
        else:
            return "DAY"
    
    def _generate_business_synonyms(self, column_name: str) -> List[str]:
        """Generate business synonyms for column"""
        
        synonyms = []
        column_lower = column_name.lower()
        
        # Check pharma synonym mappings
        for business_term, technical_terms in self.pharma_synonyms.items():
            if any(term in column_lower for term in technical_terms):
                synonyms.append(business_term)
        
        # Add common synonyms based on patterns
        if 'patient' in column_lower:
            synonyms.extend(['member', 'enrollee', 'subject'])
        elif 'provider' in column_lower or 'hcp' in column_lower:
            synonyms.extend(['physician', 'doctor', 'prescriber'])
        elif 'product' in column_lower:
            synonyms.extend(['drug', 'brand', 'medication'])
        
        return list(set(synonyms))
    
    def _classify_pii_risk(self, column_name: str) -> str:
        """Classify PII/PHI risk level"""
        
        column_lower = column_name.lower()
        
        # High risk PII/PHI
        high_risk_patterns = [
            'ssn', 'social_security', 'patient_name', 'name', 'address', 
            'phone', 'email', 'dob', 'birth_date'
        ]
        
        if any(pattern in column_lower for pattern in high_risk_patterns):
            return "HIGH"
        
        # Medium risk
        medium_risk_patterns = [
            'patient_id', 'member_id', 'zip', 'postal_code', 'mrn'
        ]
        
        if any(pattern in column_lower for pattern in medium_risk_patterns):
            return "MEDIUM"
        
        # Low risk (demographic without direct identifiers)
        low_risk_patterns = [
            'age_group', 'gender', 'state', 'region'
        ]
        
        if any(pattern in column_lower for pattern in low_risk_patterns):
            return "LOW"
        
        return "NONE"
    
    async def _discover_relationships(self, tables: Dict[str, TableInfo]) -> List[RelationshipInfo]:
        """Auto-discover table relationships"""
        
        relationships = []
        table_names = list(tables.keys())
        
        # Common join patterns in pharma data
        join_patterns = [
            {
                "from_suffix": "_id",
                "to_table_pattern": r".*{base_name}.*",
                "confidence": 0.8
            },
            {
                "from_suffix": "_key", 
                "to_table_pattern": r".*{base_name}.*",
                "confidence": 0.7
            }
        ]
        
        # Analyze each table pair
        for i, table1 in enumerate(table_names):
            for j, table2 in enumerate(table_names[i+1:], i+1):
                
                # Skip same table
                if table1 == table2:
                    continue
                
                # Find potential relationships
                potential_joins = await self._find_potential_joins(
                    tables[table1], tables[table2]
                )
                
                relationships.extend(potential_joins)
        
        # Sort by confidence and return top relationships
        relationships.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return relationships[:50]  # Limit to top 50 relationships
    
    async def _find_potential_joins(
        self, 
        table1: TableInfo, 
        table2: TableInfo
    ) -> List[RelationshipInfo]:
        """Find potential joins between two tables"""
        
        potential_joins = []
        
        # Get column names for both tables (simplified - in real implementation, 
        # we'd query the actual column metadata)
        
        # For now, use common pharmaceutical join patterns
        pharma_join_patterns = [
            {
                "column_patterns": ["patient_id", "member_id", "person_id"],
                "join_type": "INNER",
                "relationship_type": "ONE_TO_MANY"
            },
            {
                "column_patterns": ["hcp_id", "provider_id", "prescriber_id"],
                "join_type": "INNER", 
                "relationship_type": "ONE_TO_MANY"
            },
            {
                "column_patterns": ["product_id", "ndc", "drug_id"],
                "join_type": "INNER",
                "relationship_type": "ONE_TO_MANY" 
            },
            {
                "column_patterns": ["date_id", "calendar_date", "activity_date"],
                "join_type": "INNER",
                "relationship_type": "MANY_TO_ONE"
            }
        ]
        
        # Simple heuristic-based relationship detection
        table1_name = table1.table_name.lower()
        table2_name = table2.table_name.lower()
        
        # Check if tables have complementary purposes
        if ('fact' in table1_name or 'transaction' in table1_name) and \
           ('dim' in table2_name or 'master' in table2_name):
            
            potential_joins.append(RelationshipInfo(
                from_table=f"{table1.database_name}.{table1.schema_name}.{table1.table_name}",
                to_table=f"{table2.database_name}.{table2.schema_name}.{table2.table_name}",
                join_type="INNER",
                join_condition="Auto-detected based on naming convention",
                relationship_type="MANY_TO_ONE", 
                confidence_score=0.6
            ))
        
        return potential_joins
    
    async def _discover_business_metrics(
        self, 
        tables: Dict[str, TableInfo], 
        columns: Dict[str, List[Dict[str, Any]]]
    ) -> List[BusinessMetric]:
        """Auto-discover common business metrics"""
        
        discovered_metrics = []
        
        # Look for transaction/fact tables
        fact_tables = [
            table_name for table_name, table_info in tables.items()
            if table_info.business_purpose in ['Fact Table', 'Prescription Tables']
        ]
        
        for table_name in fact_tables:
            table_columns = columns.get(table_name, [])
            
            # Look for metric opportunities
            for metric_name, metric_config in self.pharma_metrics.items():
                
                # Check if this table can support the metric
                if self._can_support_metric(metric_name, table_columns):
                    
                    discovered_metrics.append(BusinessMetric(
                        metric_name=metric_name,
                        metric_type=self._classify_metric_type(metric_config['calculation']),
                        base_table=table_name,
                        calculation_logic=metric_config['calculation'],
                        grain=self._determine_metric_grain(table_columns),
                        typical_filters=self._suggest_metric_filters(table_columns),
                        business_definition=metric_config['definition']
                    ))
        
        return discovered_metrics
    
    def _can_support_metric(self, metric_name: str, table_columns: List[Dict[str, Any]]) -> bool:
        """Check if table can support a specific metric"""
        
        column_names = [col['column_name'].lower() for col in table_columns]
        
        metric_requirements = {
            'NBRx': ['prescription_id', 'first_fill'],
            'TRx': ['prescription_id'],
            'Writers': ['prescriber_id'],
            'Market_Share': ['brand_rx', 'total_class_rx'],
            'Persistence': ['patient_id', 'therapy_start_date']
        }
        
        required_columns = metric_requirements.get(metric_name, [])
        
        # Check if any required columns exist (fuzzy matching)
        for req_col in required_columns:
            if not any(req_col in col_name for col_name in column_names):
                return False
        
        return True
    
    def _classify_metric_type(self, calculation: str) -> str:
        """Classify metric type from calculation logic"""
        
        calc_lower = calculation.lower()
        
        if 'count' in calc_lower:
            return 'COUNT'
        elif 'sum' in calc_lower:
            return 'SUM'
        elif 'avg' in calc_lower:
            return 'AVG'
        elif '/' in calc_lower:
            return 'RATIO'
        else:
            return 'CUSTOM'
    
    def _determine_metric_grain(self, table_columns: List[Dict[str, Any]]) -> str:
        """Determine the natural grain of metrics from this table"""
        
        column_names = [col['column_name'].lower() for col in table_columns]
        
        if any('patient' in col for col in column_names):
            return 'Patient-level'
        elif any('hcp' in col or 'provider' in col for col in column_names):
            return 'HCP-level'
        elif any('product' in col or 'brand' in col for col in column_names):
            return 'Product-level'
        else:
            return 'Transaction-level'
    
    def _suggest_metric_filters(self, table_columns: List[Dict[str, Any]]) -> List[str]:
        """Suggest typical filters for metrics"""
        
        suggested_filters = []
        column_names = [col['column_name'].lower() for col in table_columns]
        
        # Date filters
        if any('date' in col for col in column_names):
            suggested_filters.append('Date Range')
        
        # Product filters  
        if any('product' in col or 'brand' in col for col in column_names):
            suggested_filters.append('Product/Brand')
        
        # Geography filters
        if any('state' in col or 'region' in col for col in column_names):
            suggested_filters.append('Geography')
        
        # Specialty filters
        if any('specialty' in col for col in column_names):
            suggested_filters.append('HCP Specialty')
        
        return suggested_filters
    
    async def _build_business_glossary(
        self, 
        columns: Dict[str, List[Dict[str, Any]]], 
        metrics: List[BusinessMetric]
    ) -> Dict[str, Any]:
        """Build comprehensive business glossary"""
        
        glossary = {
            "terms": {},
            "synonyms": {},
            "calculations": {},
            "abbreviations": {}
        }
        
        # Build terms from columns
        for table_name, table_columns in columns.items():
            for column in table_columns:
                business_name = column.get('business_name', column['column_name'])
                
                glossary["terms"][business_name] = {
                    "technical_name": column['column_name'],
                    "definition": column.get('description', f"Data field: {business_name}"),
                    "data_type": column['data_type'],
                    "source_table": table_name,
                    "synonyms": column.get('business_synonyms', [])
                }
                
                # Add synonyms
                for synonym in column.get('business_synonyms', []):
                    glossary["synonyms"][synonym] = business_name
        
        # Add metrics
        for metric in metrics:
            glossary["calculations"][metric.metric_name] = {
                "definition": metric.business_definition,
                "calculation": metric.calculation_logic,
                "base_table": metric.base_table,
                "type": metric.metric_type
            }
        
        # Add common pharmaceutical abbreviations
        glossary["abbreviations"].update({
            "NBRx": "New Brand Prescriptions",
            "TRx": "Total Prescriptions", 
            "HCP": "Healthcare Provider",
            "MSL": "Medical Science Liaison",
            "NDC": "National Drug Code",
            "PHI": "Protected Health Information",
            "PII": "Personally Identifiable Information"
        })
        
        return glossary
    
    def _determine_data_freshness(
        self, 
        table_name: str, 
        last_updated: Optional[datetime]
    ) -> str:
        """Determine data refresh frequency"""
        
        table_lower = table_name.lower()
        
        # Real-time indicators
        if any(word in table_lower for word in ['stream', 'real_time', 'live']):
            return "REAL_TIME"
        
        # Daily indicators
        elif any(word in table_lower for word in ['daily', 'day']):
            return "DAILY"
        
        # Weekly indicators
        elif any(word in table_lower for word in ['weekly', 'week']):
            return "WEEKLY"
        
        # Monthly indicators
        elif any(word in table_lower for word in ['monthly', 'month']):
            return "MONTHLY"
        
        # Infer from last updated
        elif last_updated:
            days_old = (datetime.now() - last_updated).days
            if days_old <= 1:
                return "DAILY"
            elif days_old <= 7:
                return "WEEKLY"
            else:
                return "MONTHLY"
        
        else:
            return "UNKNOWN"

# Create global instance
auto_discovery_schema = AutoDiscoverySchema({
    'user': 'your_user',
    'password': 'your_password',
    'account': 'your_account',
    'warehouse': 'COMPUTE_WH',
    'database': 'COMMERCIAL_AI',
    'schema': 'ENHANCED_NBA'
})
