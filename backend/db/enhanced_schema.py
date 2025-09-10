"""
Enhanced Schema System for NL2SQL Agent
Implements rich metadata schema with business context, guardrails, and semantic annotations
Based on battle-tested patterns for healthcare/life-sciences applications
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from .engine import get_adapter
from .database_documentation import get_database_documentation, format_database_documentation_for_llm, get_database_specific_examples


@dataclass
class ColumnMetadata:
    """Rich column metadata with semantic annotations"""
    name: str
    data_type: str
    nullable: bool = True
    default: Optional[str] = None
    description: Optional[str] = None
    
    # Semantic annotations
    semantic_role: Optional[str] = None  # e.g., "id", "measure", "date", "category", "geo", "text"
    units: Optional[str] = None
    currency: Optional[str] = None
    
    # Value intelligence
    min_value: Optional[Union[str, int, float]] = None
    max_value: Optional[Union[str, int, float]] = None
    p50_value: Optional[Union[str, int, float]] = None
    p95_value: Optional[Union[str, int, float]] = None
    cardinality: Optional[int] = None
    null_rate: Optional[float] = None
    top_values: Optional[List[List]] = None  # [[value, frequency], ...]
    
    # Business context
    nl_aliases: Optional[List[str]] = None
    business_definition: Optional[str] = None
    
    # Governance
    sensitivity_label: Optional[str] = None  # "phi", "pii", "confidential", "public"
    expose_in_output: bool = True
    safe_aggregation_only: bool = False
    
    # Relationships
    foreign_key_ref: Optional[str] = None  # "schema.table.column"
    is_primary_key: bool = False


@dataclass
class TableMetadata:
    """Rich table metadata with business context"""
    # Required fields first
    name: str
    full_qualified_name: str
    columns: List[ColumnMetadata]
    
    # Optional fields with defaults
    table_type: str = "table"  # "table", "view", "materialized_view"
    row_count: Optional[int] = None
    
    # Freshness
    last_loaded_at: Optional[str] = None
    last_modified_at: Optional[str] = None
    sla_minutes: Optional[int] = None
    data_latency_hours: Optional[int] = None
    
    # Performance hints
    partitions: Optional[Dict[str, str]] = None  # {"column": "FILL_DATE"}
    cluster_by: Optional[List[str]] = None
    indexes: Optional[List[str]] = None
    sample_first: bool = False
    estimated_cost: Optional[str] = None
    
    # Business context
    description: Optional[str] = None
    nl_aliases: Optional[List[str]] = None
    business_purpose: Optional[str] = None
    source_system: Optional[str] = None
    
    # Governance
    sensitivity_level: Optional[str] = None
    suppress_small_cells: bool = False
    small_cell_threshold: int = 11
    allowed_aggregation_levels: Optional[List[str]] = None
    
    # Relationships
    primary_keys: Optional[List[str]] = None
    foreign_keys: Optional[List[Dict]] = None  # [{"columns": ["col"], "ref": "schema.table.col"}]
    typical_joins: Optional[List[str]] = None


@dataclass
class KPIDefinition:
    """Business KPI with canonical definition"""
    name: str
    definition: str
    tables: List[str]
    filters: Optional[Dict] = None
    calculation: Optional[str] = None
    business_context: Optional[str] = None


@dataclass
class HierarchyDefinition:
    """Business hierarchy mapping"""
    name: str
    levels: List[str]  # ["ZIP5", "ZIP3", "County", "DMA", "Region"]
    table: str
    key_columns: Dict[str, str]  # {"ZIP5": "zip_code", "ZIP3": "zip3_code"}


@dataclass
class EnhancedSchemaSnapshot:
    """Complete schema snapshot with rich metadata"""
    engine: str
    database: str
    allowed_schemas: List[str]
    tables: List[TableMetadata]
    
    # Global guardrails
    default_limit: int = 5000
    small_cell_threshold: int = 11
    max_query_time_seconds: int = 300
    write_disabled: bool = True
    
    # Optional schema content
    kpis: Optional[List[KPIDefinition]] = None
    hierarchies: Optional[List[HierarchyDefinition]] = None
    
    # Database-specific documentation
    database_documentation: Optional[str] = None
    syntax_examples: Optional[List[Dict[str, str]]] = None
    
    # Metadata
    generated_at: str = None
    schema_version: str = "2.0"
    
    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.utcnow().isoformat() + "Z"


class EnhancedSchemaBuilder:
    """Builds enhanced schema with rich metadata from database introspection"""
    
    def __init__(self):
        self.adapter = get_adapter()
        self.cache_path = "backend/db/enhanced_schema_cache.json"
        self.cache_ttl_hours = 24  # Cache for 24 hours
        
        # Cache for bulk key detection
        self._all_constraints_cache = None
    
    def _get_all_constraints_bulk(self):
        """Get all constraints in bulk for faster key detection"""
        if self._all_constraints_cache is not None:
            return self._all_constraints_cache
            
        try:
            # Use INFORMATION_SCHEMA to get all constraints at once
            sql = """
            SELECT 
                tc.table_name,
                tc.constraint_name,
                tc.constraint_type,
                kcu.column_name,
                kcu.referenced_table_name,
                kcu.referenced_column_name
            FROM information_schema.table_constraints tc
            LEFT JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            WHERE tc.table_schema = CURRENT_SCHEMA()
            ORDER BY tc.table_name, tc.constraint_name, kcu.ordinal_position
            """
            
            result = self.adapter.run(sql)
            self._all_constraints_cache = result.rows
            return self._all_constraints_cache
            
        except Exception as e:
            print(f"Failed to get bulk constraints: {e}")
            return []
    
    def _has_valid_cache(self) -> bool:
        """Check if cache exists and is not expired"""
        try:
            if not os.path.exists(self.cache_path):
                return False
            
            # Check if cache is expired
            cache_mtime = os.path.getmtime(self.cache_path)
            cache_age = time.time() - cache_mtime
            cache_age_hours = cache_age / 3600
            
            return cache_age_hours < self.cache_ttl_hours
        except:
            return False
    
    def _load_from_cache(self) -> dict:
        """Load schema from cache"""
        with open(self.cache_path, 'r') as f:
            return json.load(f)
    
    def _save_to_cache(self, schema: 'EnhancedSchemaSnapshot') -> None:
        """Save schema to cache"""
        try:
            cache_data = {
                "schema_version": "2.0",
                "engine": schema.engine,
                "database": schema.database,
                "allowed_schemas": schema.allowed_schemas,
                "default_limit": schema.default_limit,
                "small_cell_threshold": schema.small_cell_threshold,
                "tables": [_table_to_dict(t) for t in schema.tables],
                "kpis": [_kpi_to_dict(k) for k in schema.kpis or []],
                "hierarchies": [_hierarchy_to_dict(h) for h in schema.hierarchies or []],
                "database_documentation": schema.database_documentation,
                "syntax_examples": schema.syntax_examples,
                "generated_at": schema.generated_at
            }
            
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Failed to save schema cache: {e}")
    
    def build_enhanced_snapshot(self, adapter=None, quick_mode=True) -> dict:
        """Build enhanced schema snapshot - convenience method for main.py and pipeline.py"""
        if adapter:
            self.adapter = adapter
        
        # Check cache first for quick startup
        if quick_mode and self._has_valid_cache():
            print("💨 Using cached schema for quick startup...")
            return self._load_from_cache()
        
        # Use default allowlist for public schema
        allowlist = ["public"]
        enhanced_schema = self.build_enhanced_schema(allowlist, quick_mode=quick_mode)
        
        # Cache the result
        if quick_mode:
            self._save_to_cache(enhanced_schema)
        
        # Convert to dict format for JSON serialization
        return {
            "schema_version": "2.0",
            "engine": enhanced_schema.engine,
            "database": enhanced_schema.database,
            "allowed_schemas": enhanced_schema.allowed_schemas,
            "default_limit": enhanced_schema.default_limit,
            "small_cell_threshold": enhanced_schema.small_cell_threshold,
            "tables": [_table_to_dict(t) for t in enhanced_schema.tables],
            "kpis": [_kpi_to_dict(k) for k in enhanced_schema.kpis or []],
            "hierarchies": [_hierarchy_to_dict(h) for h in enhanced_schema.hierarchies or []],
            "database_documentation": enhanced_schema.database_documentation,
            "syntax_examples": enhanced_schema.syntax_examples,
            "generated_at": enhanced_schema.generated_at
        }
        
    def build_enhanced_schema(self, allowlist: List[str], quick_mode=True) -> EnhancedSchemaSnapshot:
        """Build complete enhanced schema with metadata"""
        if quick_mode:
            print("💨 Building enhanced schema (quick mode)...")
        else:
            print("🔧 Building enhanced schema with rich metadata...")
        
        # Get basic schema from adapter
        basic_schema = self.adapter.get_schema_snapshot(allowlist)
        
        # Initialize enhanced schema
        engine_type = self._detect_engine_type()
        enhanced_schema = EnhancedSchemaSnapshot(
            engine=engine_type,
            database=self._get_database_name(),
            allowed_schemas=allowlist,
            tables=[]
        )
        
        # Add database-specific documentation
        print(f"  📚 Adding {engine_type} documentation and syntax guide...")
        dialect = get_database_documentation(engine_type)
        enhanced_schema.database_documentation = format_database_documentation_for_llm(dialect)
        enhanced_schema.syntax_examples = get_database_specific_examples(engine_type)
        
        # Pre-load all constraints for bulk processing (optimization)
        print("🔍 Pre-loading relationship constraints...")
        self._get_all_constraints_bulk()
        
        # Build enhanced table metadata (parallel processing for speed)
        print(f"🚀 Analyzing {len(basic_schema)} tables in parallel...")
        
        if len(basic_schema) > 1 and quick_mode:
            # Use parallel processing for multiple tables in quick mode
            with ThreadPoolExecutor(max_workers=min(4, len(basic_schema))) as executor:
                future_to_table = {
                    executor.submit(self._build_table_metadata, table_name, columns, quick_mode): table_name 
                    for table_name, columns in basic_schema.items()
                }
                
                for future in as_completed(future_to_table):
                    table_meta = future.result()
                    enhanced_schema.tables.append(table_meta)
        else:
            # Sequential processing for single table or full mode
            for table_name, columns in basic_schema.items():
                table_meta = self._build_table_metadata(table_name, columns, quick_mode)
                enhanced_schema.tables.append(table_meta)
        
        # Add business KPIs and hierarchies
        enhanced_schema.kpis = self._get_business_kpis()
        enhanced_schema.hierarchies = self._get_business_hierarchies()
        
        return enhanced_schema
    
    def _build_table_metadata(self, table_name: str, columns: Dict[str, str], quick_mode=True) -> TableMetadata:
        """Build rich table metadata with introspection"""
        print(f"  📊 Analyzing table: {table_name}")
        
        # Get table stats (skip in quick mode)
        if quick_mode:
            row_count = None
            freshness_info = {}
        else:
            row_count = self._get_table_row_count(table_name)
            freshness_info = self._get_table_freshness(table_name)
        
        # Build column metadata
        enhanced_columns = []
        for col_name, col_type in columns.items():
            col_meta = self._build_column_metadata(table_name, col_name, col_type, quick_mode)
            enhanced_columns.append(col_meta)
        
        # Detect relationships (essential for complex queries even in quick mode)
        primary_keys = self._detect_primary_keys(table_name)
        foreign_keys = self._detect_foreign_keys(table_name)
        
        # NEW: Semantic relationship detection when formal constraints don't exist
        if not primary_keys and not foreign_keys:
            semantic_relationships = self._detect_semantic_relationships(table_name, columns)
            primary_keys = semantic_relationships.get('primary_keys')
            foreign_keys = semantic_relationships.get('foreign_keys')
        
        # Apply business rules and governance
        table_meta = TableMetadata(
            name=table_name,
            full_qualified_name=f"public.{table_name}",  # Adjust based on schema
            row_count=row_count,
            columns=enhanced_columns,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            **(freshness_info if not quick_mode else {})
        )
        
        # Apply domain-specific enhancements
        self._apply_domain_enhancements(table_meta)
        
        return table_meta
    
    def _build_column_metadata(self, table_name: str, col_name: str, col_type: str, quick_mode=True) -> ColumnMetadata:
        """Build rich column metadata with semantic analysis"""
        
        # Basic metadata
        col_meta = ColumnMetadata(
            name=col_name,
            data_type=col_type,
            nullable=True,  # Default to True in quick mode
        )
        
        # Semantic role detection (fast)
        col_meta.semantic_role = self._detect_semantic_role(col_name, col_type)
        
        # Skip heavy value analysis in quick mode
        if not quick_mode and self._should_analyze_values(col_type):
            value_stats = self._analyze_column_values(table_name, col_name, col_type)
            col_meta.min_value = value_stats.get('min')
            col_meta.max_value = value_stats.get('max')
            col_meta.cardinality = value_stats.get('cardinality')
            col_meta.null_rate = value_stats.get('null_rate')
            col_meta.top_values = value_stats.get('top_values')
        else:
            # Set defaults for quick mode
            col_meta.cardinality = None
            col_meta.null_rate = None
        
        # Business aliases and governance
        col_meta.nl_aliases = self._get_column_aliases(table_name, col_name)
        col_meta.sensitivity_label = self._detect_sensitivity(col_name)
        col_meta.expose_in_output = self._check_output_exposure(col_name)
        
        return col_meta
    
    def _detect_semantic_role(self, col_name: str, col_type: str) -> str:
        """Detect semantic role of column based on name and type"""
        col_lower = col_name.lower()
        
        # ID patterns
        if col_lower.endswith('_id') or col_lower.startswith('id_') or col_lower == 'id':
            return "entity.id"
        
        # Date patterns  
        if 'date' in col_lower or 'time' in col_lower or col_type.upper() in ['DATE', 'DATETIME', 'TIMESTAMP']:
            if any(word in col_lower for word in ['created', 'modified', 'updated', 'loaded']):
                return "date.system"
            return "date.event"
        
        # Measure patterns
        if any(word in col_lower for word in ['count', 'amount', 'quantity', 'total', 'sum', 'avg']):
            return "measure.quantity"
        
        if any(word in col_lower for word in ['price', 'cost', 'revenue', 'value']):
            return "measure.currency"
        
        # Geographic patterns
        if any(word in col_lower for word in ['zip', 'postal', 'lat', 'lon', 'geo', 'region', 'state']):
            return "geo.location"
        
        # Category patterns
        if any(word in col_lower for word in ['type', 'category', 'status', 'group', 'class']):
            return "category"
        
        # Text patterns
        if col_type.upper() in ['TEXT', 'VARCHAR', 'STRING'] and 'name' in col_lower:
            return "text.name"
        
        return "other"
    
    def _detect_sensitivity(self, col_name: str) -> str:
        """Detect data sensitivity level"""
        col_lower = col_name.lower()
        
        # PHI patterns
        if any(word in col_lower for word in ['patient', 'ssn', 'social', 'dob', 'birth', 'medical']):
            return "phi"
        
        # PII patterns  
        if any(word in col_lower for word in ['email', 'phone', 'address', 'name']):
            return "pii"
        
        # Pseudonymized patterns
        if any(word in col_lower for word in ['hash', 'encrypted', 'masked']):
            return "pseudonymized"
        
        return "public"
    
    def _check_output_exposure(self, col_name: str) -> bool:
        """Check if column should be exposed in output"""
        col_lower = col_name.lower()
        
        # Never expose these patterns
        sensitive_patterns = ['ssn', 'social_security', 'patient_id', 'medical_record']
        return not any(pattern in col_lower for pattern in sensitive_patterns)
    
    def _get_column_aliases(self, table_name: str, col_name: str) -> List[str]:
        """Get natural language aliases for column"""
        aliases = []
        col_lower = col_name.lower()
        
        # Common business aliases
        alias_map = {
            'rx_type': ['prescription type', 'trx type', 'transaction type'],
            'hcp_id': ['physician id', 'healthcare provider', 'doctor id'],
            'patient_hash': ['patient', 'member id', 'patient identifier'],
            'fill_date': ['prescription date', 'fill date', 'dispensing date'],
            'quantity': ['qty', 'amount dispensed', 'units'],
            'product_id': ['drug id', 'medication id', 'ndc'],
        }
        
        return alias_map.get(col_lower, [])
    
    def _apply_domain_enhancements(self, table_meta: TableMetadata):
        """Apply domain-specific business rules and metadata"""
        table_lower = table_meta.name.lower()
        
        # Healthcare/Pharma specific rules
        if any(term in table_lower for term in ['rx', 'prescription', 'claim', 'fill']):
            table_meta.business_purpose = "Healthcare transaction data"
            table_meta.suppress_small_cells = True
            table_meta.small_cell_threshold = 11
            table_meta.nl_aliases = ['prescriptions', 'claims', 'fills', 'transactions']
        
        elif any(term in table_lower for term in ['patient', 'member']):
            table_meta.business_purpose = "Patient/member information"
            table_meta.sensitivity_level = "phi"
            table_meta.suppress_small_cells = True
        
        elif any(term in table_lower for term in ['hcp', 'physician', 'doctor']):
            table_meta.business_purpose = "Healthcare provider information"
            table_meta.nl_aliases = ['doctors', 'physicians', 'providers', 'hcps']
        
        elif any(term in table_lower for term in ['product', 'drug', 'medication']):
            table_meta.business_purpose = "Product/medication reference data"
            table_meta.nl_aliases = ['drugs', 'medications', 'products', 'therapies']
    
    def _get_business_kpis(self) -> List[KPIDefinition]:
        """Define standard business KPIs"""
        return [
            KPIDefinition(
                name="nps",
                definition="New Patient Starts - count of unique patients starting therapy",
                tables=["rx_facts", "prescription_data"],
                filters={"rx_type": "NEW_START"},
                business_context="Key metric for measuring therapy adoption"
            ),
            KPIDefinition(
                name="trx",
                definition="Total prescriptions including new and refills",
                tables=["rx_facts", "prescription_data"],
                calculation="COUNT(*)",
                business_context="Volume metric for prescription activity"
            ),
            KPIDefinition(
                name="persistence_6mo", 
                definition="Patients continuing therapy for 6+ months",
                tables=["rx_facts"],
                calculation="COUNT(DISTINCT patient_id) with fills spanning 6+ months",
                business_context="Adherence and treatment continuation metric"
            )
        ]
    
    def _get_business_hierarchies(self) -> List[HierarchyDefinition]:
        """Define standard business hierarchies"""
        return [
            HierarchyDefinition(
                name="geography",
                levels=["ZIP5", "ZIP3", "County", "DMA", "Region"],
                table="geo_hierarchy",
                key_columns={
                    "ZIP5": "zip_code",
                    "ZIP3": "zip3_code", 
                    "County": "county_name",
                    "DMA": "dma_code",
                    "Region": "region_name"
                }
            ),
            HierarchyDefinition(
                name="product",
                levels=["NDC", "Brand", "Generic", "Therapeutic_Area"],
                table="product_hierarchy",
                key_columns={
                    "NDC": "ndc_code",
                    "Brand": "brand_name",
                    "Generic": "generic_name", 
                    "Therapeutic_Area": "therapeutic_area"
                }
            )
        ]
    
    # Helper methods for database introspection
    def _detect_engine_type(self) -> str:
        """Detect database engine type"""
        if hasattr(self.adapter, 'config') and 'snowflake' in str(type(self.adapter)).lower():
            return "snowflake"
        elif 'postgres' in str(type(self.adapter)).lower():
            return "postgresql"
        return "unknown"
    
    def _get_database_name(self) -> str:
        """Get database name"""
        return os.getenv("DATABASE_NAME", "analytics")
    
    def _get_table_row_count(self, table_name: str) -> Optional[int]:
        """Get approximate row count for table"""
        try:
            result = self.adapter.run(f"SELECT COUNT(*) FROM {table_name} LIMIT 1")
            if result.rows:
                return result.rows[0][0]
        except:
            pass
        return None
    
    def _get_table_freshness(self, table_name: str) -> Dict:
        """Get table freshness information"""
        return {
            "last_loaded_at": (datetime.utcnow() - timedelta(hours=2)).isoformat() + "Z",
            "sla_minutes": 1440,  # 24 hours
            "data_latency_hours": 2
        }
    
    def _detect_primary_keys(self, table_name: str) -> Optional[List[str]]:
        """Detect primary key columns using bulk constraint cache"""
        try:
            constraints = self._get_all_constraints_bulk()
            primary_keys = []
            
            for row in constraints:
                if (row[0] == table_name and 
                    row[2] == 'PRIMARY KEY' and 
                    row[3] is not None):
                    primary_keys.append(row[3])
            
            return primary_keys if primary_keys else None
        except Exception as e:
            print(f"Failed to detect primary keys for {table_name}: {e}")
            return None
    
    def _detect_foreign_keys(self, table_name: str) -> Optional[List[Dict]]:
        """Detect foreign key relationships using bulk constraint cache"""
        try:
            constraints = self._get_all_constraints_bulk()
            foreign_keys = []
            
            for row in constraints:
                if (row[0] == table_name and 
                    row[2] == 'FOREIGN KEY' and 
                    row[3] is not None and 
                    row[4] is not None and 
                    row[5] is not None):
                    
                    foreign_keys.append({
                        "column": row[3],
                        "referenced_table": row[4],
                        "referenced_column": row[5]
                    })
            
            return foreign_keys if foreign_keys else None
        except Exception as e:
            print(f"Failed to detect foreign keys for {table_name}: {e}")
            return None

    def _detect_semantic_relationships(self, table_name: str, columns: Dict[str, str]) -> Dict:
        """Detect semantic relationships when formal PK/FK constraints don't exist"""
        try:
            print(f"🔍 Analyzing semantic relationships for {table_name}")
            
            # Get all tables and their columns for cross-table analysis
            all_tables_info = self._get_all_tables_columns()
            
            semantic_primary_keys = []
            semantic_foreign_keys = []
            
            # 1. Identify likely primary keys based on naming patterns
            for col_name, col_type in columns.items():
                if self._is_likely_primary_key(col_name, col_type, table_name):
                    semantic_primary_keys.append(col_name)
                    print(f"  🔑 Identified semantic PK: {table_name}.{col_name}")
            
            # 2. Find potential foreign key relationships
            for col_name, col_type in columns.items():
                potential_refs = self._find_potential_foreign_keys(
                    col_name, col_type, table_name, all_tables_info
                )
                for ref in potential_refs:
                    semantic_foreign_keys.append({
                        "column": col_name,
                        "referenced_table": ref["table"],
                        "referenced_column": ref["column"],
                        "confidence": ref["confidence"],
                        "match_type": ref["match_type"]
                    })
                    print(f"  🔗 Semantic FK: {table_name}.{col_name} → {ref['table']}.{ref['column']} ({ref['confidence']:.2f})")
            
            return {
                "primary_keys": semantic_primary_keys if semantic_primary_keys else None,
                "foreign_keys": semantic_foreign_keys if semantic_foreign_keys else None
            }
            
        except Exception as e:
            print(f"Failed to detect semantic relationships for {table_name}: {e}")
            return {"primary_keys": None, "foreign_keys": None}
    
    def _get_all_tables_columns(self) -> Dict[str, Dict[str, str]]:
        """Get all tables and their columns for relationship analysis"""
        try:
            tables_info = {}
            
            # Get all tables
            result = self.adapter.run("SHOW TABLES")
            if result.error:
                return {}
                
            for row in result.rows:
                table_name = row[1] if len(row) > 1 else row[0]
                
                # Get columns for each table
                col_result = self.adapter.run(f'DESCRIBE TABLE "{table_name}"')
                if not col_result.error and col_result.rows:
                    tables_info[table_name] = {}
                    for col_row in col_result.rows:
                        col_name = col_row[0]
                        col_type = col_row[1]
                        tables_info[table_name][col_name] = col_type
            
            return tables_info
            
        except Exception as e:
            print(f"Failed to get all tables columns: {e}")
            return {}
    
    def _is_likely_primary_key(self, col_name: str, col_type: str, table_name: str) -> bool:
        """Determine if a column is likely a primary key based on naming patterns"""
        col_name_lower = col_name.lower()
        table_name_lower = table_name.lower()
        
        # Common primary key patterns
        pk_patterns = [
            'id',
            f'{table_name_lower}_id',
            f'{table_name_lower.rstrip("s")}_id',  # Remove trailing 's'
            'uuid',
            'guid',
            'key'
        ]
        
        # Healthcare-specific patterns
        healthcare_id_patterns = [
            'npi',  # National Provider Identifier
            'provider_id',
            'patient_id',
            'member_id',
            'account_id',
            'group_id'
        ]
        
        pk_patterns.extend(healthcare_id_patterns)
        
        # Check exact matches
        if col_name_lower in pk_patterns:
            return True
            
        # Check if column name ends with common PK suffixes
        pk_suffixes = ['_id', '_key', '_uuid', '_guid']
        if any(col_name_lower.endswith(suffix) for suffix in pk_suffixes):
            return True
            
        return False
    
    def _find_potential_foreign_keys(self, col_name: str, col_type: str, 
                                   source_table: str, all_tables: Dict[str, Dict[str, str]]) -> List[Dict]:
        """Find potential foreign key relationships for a column"""
        potential_refs = []
        col_name_lower = col_name.lower()
        
        for target_table, target_columns in all_tables.items():
            if target_table == source_table:
                continue
                
            for target_col, target_type in target_columns.items():
                confidence = self._calculate_relationship_confidence(
                    col_name, col_type, target_col, target_type, source_table, target_table
                )
                
                if confidence > 0.5:  # Only include high-confidence matches
                    match_type = self._get_match_type(col_name, target_col, col_type, target_type)
                    potential_refs.append({
                        "table": target_table,
                        "column": target_col,
                        "confidence": confidence,
                        "match_type": match_type
                    })
        
        # Sort by confidence (highest first)
        return sorted(potential_refs, key=lambda x: x["confidence"], reverse=True)
    
    def _calculate_relationship_confidence(self, col1: str, type1: str, col2: str, type2: str,
                                         table1: str, table2: str) -> float:
        """Calculate confidence score for potential relationship using multiple signals"""
        col1_lower = col1.lower()
        col2_lower = col2.lower()
        
        # 1. Data type compatibility (essential)
        if not self._are_types_compatible(type1, type2):
            return 0.0  # No relationship if types incompatible
        
        # 2. Pattern-based confidence (baseline)
        pattern_confidence = 0.0
        
        # Exact column name match (very high confidence)
        if col1_lower == col2_lower:
            pattern_confidence = 0.9
        # Similar naming patterns
        elif self._are_names_similar(col1_lower, col2_lower):
            pattern_confidence = 0.7
        else:
            # Healthcare domain-specific patterns
            healthcare_confidence = self._get_healthcare_relationship_confidence(
                col1_lower, col2_lower, table1, table2
            )
            # Table relationship patterns
            table_confidence = self._get_table_relationship_confidence(table1, table2)
            pattern_confidence = healthcare_confidence + table_confidence
        
        # 3. Data-driven analysis (stronger signal)
        data_confidence = self._get_data_driven_confidence(table1, col1, table2, col2)
        
        # 4. Combine signals with weighted approach
        if data_confidence > 0:
            # Data analysis gets higher weight when available
            final_confidence = (pattern_confidence * 0.4) + (data_confidence * 0.6)
        else:
            # Fallback to pattern matching only
            final_confidence = pattern_confidence
            print(f"📊 Using pattern-only for {table1}.{col1} ↔ {table2}.{col2}: {final_confidence:.2f}")
        
        return min(final_confidence, 1.0)  # Cap at 1.0
    
    def _get_data_driven_confidence(self, table1: str, col1: str, table2: str, col2: str) -> float:
        """Analyze actual data to determine relationship confidence"""
        try:
            print(f"🔍 Analyzing data for {table1}.{col1} ↔ {table2}.{col2}")
            
            # Sample data from both columns
            sample_size = 500  # Reasonable sample size
            sample1 = self._get_column_sample(table1, col1, sample_size)
            sample2 = self._get_column_sample(table2, col2, sample_size)
            
            if not sample1 or not sample2:
                return 0.0
            
            # Calculate different types of similarity
            overlap_score = self._calculate_value_overlap(sample1, sample2)
            cardinality_score = self._calculate_cardinality_score(sample1, sample2)
            format_score = self._calculate_format_similarity(sample1, sample2)
            
            # Weighted combination
            data_confidence = (overlap_score * 0.5) + (cardinality_score * 0.3) + (format_score * 0.2)
            
            if data_confidence > 0.1:  # Only log significant findings
                print(f"📈 Data analysis: overlap={overlap_score:.2f}, cardinality={cardinality_score:.2f}, format={format_score:.2f} → {data_confidence:.2f}")
            
            return data_confidence
            
        except Exception as e:
            # Graceful fallback - don't break the system
            print(f"⚠️  Data analysis failed for {table1}.{col1} ↔ {table2}.{col2}: {e}")
            return 0.0

    def _get_column_sample(self, table: str, column: str, sample_size: int = 1000) -> list:
        """Safely sample data from a column for analysis"""
        try:
            # Use SAMPLE or LIMIT depending on database capabilities
            sql = f"""
            SELECT DISTINCT {column} 
            FROM {table} 
            WHERE {column} IS NOT NULL 
            LIMIT {sample_size}
            """
            
            result = self.adapter.run(sql)
            if result.error:
                return []
            
            # Extract values from result rows
            values = [row[0] for row in result.rows if row[0] is not None]
            return values
            
        except Exception as e:
            print(f"⚠️  Error sampling {table}.{column}: {e}")
            return []
    
    def _calculate_value_overlap(self, sample1: list, sample2: list) -> float:
        """Calculate the overlap percentage between two value sets"""
        try:
            if not sample1 or not sample2:
                return 0.0
            
            set1 = set(str(v).strip().lower() for v in sample1 if v is not None)
            set2 = set(str(v).strip().lower() for v in sample2 if v is not None)
            
            if not set1 or not set2:
                return 0.0
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            if union == 0:
                return 0.0
            
            # Jaccard similarity coefficient
            overlap_score = intersection / union
            
            # Add JOIN safety warning for low overlap
            if intersection == 0:
                print(f"⚠️  JOIN WARNING: No overlapping values found - JOIN may return empty results")
            elif overlap_score < 0.1:
                print(f"⚠️  JOIN CAUTION: Very low overlap ({overlap_score:.1%}) - JOIN may return sparse results")
            
            # High overlap suggests FK relationship
            if overlap_score > 0.7:
                return 0.9
            elif overlap_score > 0.3:
                return 0.6
            elif overlap_score > 0.1:
                return 0.3
            else:
                return 0.0
                
        except Exception as e:
            print(f"⚠️  Error calculating value overlap: {e}")
            return 0.0
    
    def _calculate_cardinality_score(self, sample1: list, sample2: list) -> float:
        """Analyze cardinality patterns to detect PK/FK relationships"""
        try:
            if not sample1 or not sample2:
                return 0.0
            
            unique1 = len(set(sample1))
            unique2 = len(set(sample2))
            total1 = len(sample1)
            total2 = len(sample2)
            
            # Calculate uniqueness ratios
            ratio1 = unique1 / total1 if total1 > 0 else 0
            ratio2 = unique2 / total2 if total2 > 0 else 0
            
            # PK-like pattern: high uniqueness (>0.9)
            # FK-like pattern: lower uniqueness (<0.8)
            
            # One column is PK-like, other is FK-like
            if (ratio1 > 0.9 and ratio2 < 0.8) or (ratio2 > 0.9 and ratio1 < 0.8):
                return 0.8
            
            # Both have similar cardinality patterns
            elif abs(ratio1 - ratio2) < 0.2:
                return 0.5
            
            return 0.0
            
        except Exception as e:
            print(f"⚠️  Error calculating cardinality: {e}")
            return 0.0
    
    def _calculate_format_similarity(self, sample1: list, sample2: list) -> float:
        """Check if values have similar formats (length, patterns)"""
        try:
            if not sample1 or not sample2:
                return 0.0
            
            # Convert to strings and get first few samples
            str1 = [str(v) for v in sample1[:100] if v is not None]
            str2 = [str(v) for v in sample2[:100] if v is not None]
            
            if not str1 or not str2:
                return 0.0
            
            # Check average length similarity
            avg_len1 = sum(len(s) for s in str1) / len(str1)
            avg_len2 = sum(len(s) for s in str2) / len(str2)
            
            length_similarity = 1 - abs(avg_len1 - avg_len2) / max(avg_len1, avg_len2, 1)
            
            # Check if they're all numeric
            numeric1 = all(s.replace('.', '').replace('-', '').isdigit() for s in str1[:10])
            numeric2 = all(s.replace('.', '').replace('-', '').isdigit() for s in str2[:10])
            
            if numeric1 and numeric2:
                return max(0.7, length_similarity)
            
            # Check pattern similarity for strings
            if length_similarity > 0.8:
                return 0.6
            elif length_similarity > 0.5:
                return 0.3
            
            return 0.0
            
        except Exception as e:
            print(f"⚠️  Error calculating format similarity: {e}")
            return 0.0
    
    def _are_types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two data types are compatible for joins"""
        # Normalize types
        type1_norm = self._normalize_data_type(type1)
        type2_norm = self._normalize_data_type(type2)
        
        # Numeric types can join with numeric types
        numeric_types = {'NUMBER', 'INTEGER', 'BIGINT', 'DECIMAL', 'FLOAT', 'DOUBLE'}
        if type1_norm in numeric_types and type2_norm in numeric_types:
            return True
            
        # Text types can join with text types
        text_types = {'VARCHAR', 'STRING', 'TEXT', 'CHAR'}
        if type1_norm in text_types and type2_norm in text_types:
            return True
            
        # Exact type match
        if type1_norm == type2_norm:
            return True
            
        return False
    
    def _normalize_data_type(self, data_type: str) -> str:
        """Normalize data type for comparison"""
        if not data_type:
            return "UNKNOWN"
            
        # Extract base type (remove size specifications)
        base_type = data_type.upper().split('(')[0].strip()
        
        # Normalize variations
        type_mappings = {
            'INT': 'INTEGER',
            'NUMERIC': 'NUMBER',
            'DECIMAL': 'NUMBER',
            'FLOAT': 'NUMBER',
            'DOUBLE': 'NUMBER',
            'CHARACTER': 'CHAR',
            'CHARACTER VARYING': 'VARCHAR',
            'STRING': 'VARCHAR'
        }
        
        return type_mappings.get(base_type, base_type)
    
    def _are_names_similar(self, name1: str, name2: str) -> bool:
        """Check if column names are similar"""
        # Remove common prefixes/suffixes for comparison
        cleaned_name1 = self._clean_column_name(name1)
        cleaned_name2 = self._clean_column_name(name2)
        
        # Exact match after cleaning
        if cleaned_name1 == cleaned_name2:
            return True
        
        # For healthcare: NPI should only match other NPI columns, not NPI_TOTAL_PCT
        if cleaned_name1 == 'npi' or cleaned_name2 == 'npi':
            # Only allow exact 'npi' matches, not partial matches like 'npi_total_pct'
            return cleaned_name1 == cleaned_name2
        
        # Check if one contains the other (but not for NPI to avoid false matches)
        if len(cleaned_name1) > 3 and len(cleaned_name2) > 3:  # Avoid short false matches
            if cleaned_name1 in cleaned_name2 or cleaned_name2 in cleaned_name1:
                return True
            
        # Check common variations for longer names
        if len(cleaned_name1) > 3:
            variations = [
                (cleaned_name1 + '_id', cleaned_name2),
                (cleaned_name1, cleaned_name2 + '_id'),
                (cleaned_name1 + '_key', cleaned_name2),
                (cleaned_name1, cleaned_name2 + '_key')
            ]
            
            for var1, var2 in variations:
                if var1 == var2:
                    return True
                
        return False
    
    def _clean_column_name(self, name: str) -> str:
        """Clean column name for comparison"""
        # Remove common suffixes
        suffixes_to_remove = ['_id', '_key', '_uuid', '_guid', '_number', '_code']
        cleaned = name.lower()
        
        for suffix in suffixes_to_remove:
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)]
                break
                
        return cleaned
    
    def _get_healthcare_relationship_confidence(self, col1: str, col2: str, table1: str, table2: str) -> float:
        """Get healthcare domain-specific relationship confidence"""
        confidence = 0.0
        
        # NPI (National Provider Identifier) relationships
        if 'npi' in col1 and 'npi' in col2:
            confidence += 0.8
        
        # Provider relationships
        if ('provider' in col1 and 'provider' in col2) or \
           ('provider' in col1 and 'provider' in table2.lower()) or \
           ('provider' in table1.lower() and 'provider' in col2):
            confidence += 0.6
        
        # Payer relationships
        if 'payer' in col1 and 'payer' in col2:
            confidence += 0.7
        
        # Service/Billing code relationships
        if ('service' in col1 and 'service' in col2) or \
           ('billing' in col1 and 'billing' in col2) or \
           ('code' in col1 and 'code' in col2):
            confidence += 0.5
        
        return min(confidence, 0.8)  # Cap healthcare bonus
    
    def _get_table_relationship_confidence(self, table1: str, table2: str) -> float:
        """Get table-level relationship confidence"""
        confidence = 0.0
        
        table1_lower = table1.lower()
        table2_lower = table2.lower()
        
        # Fact-dimension patterns (healthcare)
        fact_tables = ['metrics', 'volume', 'rates', 'claims', 'transactions']
        dimension_tables = ['provider', 'payer', 'service', 'geography', 'time']
        
        is_fact1 = any(fact in table1_lower for fact in fact_tables)
        is_dim2 = any(dim in table2_lower for dim in dimension_tables)
        
        if is_fact1 and is_dim2:
            confidence += 0.3
        
        # Reference table patterns
        if 'reference' in table2_lower or 'ref' in table2_lower:
            confidence += 0.2
        
        return confidence
    
    def _get_match_type(self, col1: str, col2: str, type1: str, type2: str) -> str:
        """Classify the type of relationship match"""
        if col1.lower() == col2.lower():
            return "exact_name_match"
        elif self._are_names_similar(col1.lower(), col2.lower()):
            return "similar_name_match"
        elif 'npi' in col1.lower() and 'npi' in col2.lower():
            return "healthcare_domain_match"
        elif self._normalize_data_type(type1) == self._normalize_data_type(type2):
            return "type_compatible_match"
        else:
            return "inferred_match"
    
    def _check_nullable(self, table_name: str, col_name: str) -> bool:
        """Check if column is nullable"""
        return True  # Default assumption
    
    def _should_analyze_values(self, col_type: str) -> bool:
        """Determine if we should analyze column values"""
        # Skip analysis for large text fields
        return col_type.upper() not in ['TEXT', 'BLOB', 'CLOB']
    
    def _analyze_column_values(self, table_name: str, col_name: str, col_type: str) -> Dict:
        """Analyze column values for intelligence (sample-based)"""
        try:
            # Sample-based analysis for performance
            if col_type.upper() in ['INTEGER', 'FLOAT', 'DECIMAL', 'NUMBER']:
                # Numeric analysis
                sql = f"""
                SELECT 
                    MIN({col_name}) as min_val,
                    MAX({col_name}) as max_val,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {col_name}) as p50,
                    COUNT(DISTINCT {col_name}) as cardinality,
                    COUNT(CASE WHEN {col_name} IS NULL THEN 1 END)::FLOAT / COUNT(*)::FLOAT as null_rate
                FROM {table_name}
                """
                result = self.adapter.run(sql)
                if result.rows:
                    row = result.rows[0]
                    return {
                        'min': row[0],
                        'max': row[1], 
                        'p50': row[2],
                        'cardinality': row[3],
                        'null_rate': row[4]
                    }
            else:
                # Categorical analysis
                sql = f"""
                SELECT {col_name}, COUNT(*) as freq
                FROM {table_name}
                WHERE {col_name} IS NOT NULL
                GROUP BY {col_name}
                ORDER BY COUNT(*) DESC
                LIMIT 10
                """
                result = self.adapter.run(sql)
                if result.rows:
                    total_rows = self._get_table_row_count(table_name) or 1
                    top_values = [[row[0], row[1] / total_rows] for row in result.rows]
                    return {
                        'top_values': top_values,
                        'cardinality': len(result.rows)
                    }
        except:
            pass
        
        return {}


def get_enhanced_schema_cache() -> Dict:
    """Get enhanced schema with rich metadata"""
    cache_path = "backend/db/enhanced_schema_cache.json"
    
    # Check if cache exists and is recent (within 1 hour)
    if os.path.exists(cache_path):
        mod_time = os.path.getmtime(cache_path)
        if time.time() - mod_time < 3600:  # 1 hour cache
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
    
    # Build fresh enhanced schema
    builder = EnhancedSchemaBuilder()
    allowlist = os.getenv("ALLOWED_SCHEMAS", "public").split(",")
    enhanced_schema = builder.build_enhanced_schema(allowlist)
    
    # Cache the result
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(asdict(enhanced_schema), f, indent=2, default=str)
    except Exception as e:
        print(f"Warning: Could not cache enhanced schema: {e}")
    
    return asdict(enhanced_schema)


def _table_to_dict(table: TableMetadata) -> dict:
    """Convert TableMetadata to dictionary"""
    return {
        "name": table.name,
        "full_qualified_name": table.full_qualified_name,
        "table_type": table.table_type,
        "row_count": table.row_count,
        "columns": [_column_to_dict(c) for c in table.columns],
        "description": table.description,
        "business_purpose": table.business_purpose,
        "sensitivity_level": table.sensitivity_level,
        "primary_keys": table.primary_keys,
        "foreign_keys": table.foreign_keys
    }

def _column_to_dict(col: ColumnMetadata) -> dict:
    """Convert ColumnMetadata to dictionary"""
    return {
        "name": col.name,
        "data_type": col.data_type,
        "nullable": col.nullable,
        "semantic_role": col.semantic_role,
        "description": col.description,
        "nl_aliases": col.nl_aliases,
        "business_definition": col.business_definition,
        "sensitivity_label": col.sensitivity_label,
        "min_value": col.min_value,
        "max_value": col.max_value,
        "cardinality": col.cardinality,
        "top_values": col.top_values
    }

def _kpi_to_dict(kpi: KPIDefinition) -> dict:
    """Convert KPIDefinition to dictionary"""
    return {
        "name": kpi.name,
        "definition": kpi.definition,
        "tables": kpi.tables,
        "calculation": kpi.calculation,
        "business_context": kpi.business_context
    }

def _hierarchy_to_dict(hierarchy: HierarchyDefinition) -> dict:
    """Convert HierarchyDefinition to dictionary"""
    return {
        "name": hierarchy.name,
        "levels": hierarchy.levels,
        "table": hierarchy.table,
        "key_columns": hierarchy.key_columns
    }


def format_schema_for_llm(enhanced_schema: Dict) -> str:
    """Format enhanced schema for LLM consumption with database-specific documentation"""
    
    # Start with database-specific documentation
    documentation_section = ""
    if enhanced_schema.get("database_documentation"):
        documentation_section = f"""
{enhanced_schema["database_documentation"]}

DATABASE-SPECIFIC EXAMPLES:
"""
        if enhanced_schema.get("syntax_examples"):
            for example in enhanced_schema["syntax_examples"]:
                documentation_section += f"""
- {example["description"]}:
  {example["sql"]}
"""
        documentation_section += "\n" + "="*80 + "\n"
    
    # Extract key information for prompt
    schema_summary = {
        "engine": enhanced_schema.get("engine"),
        "allowed_schemas": enhanced_schema.get("allowed_schemas"),
        "guardrails": {
            "default_limit": enhanced_schema.get("default_limit"),
            "small_cell_threshold": enhanced_schema.get("small_cell_threshold"),
            "write_disabled": enhanced_schema.get("write_disabled")
        },
        "tables": []
    }
    
    # Format tables with essential metadata
    for table in enhanced_schema.get("tables", []):
        table_info = {
            "name": table["name"],
            "fqtn": table["full_qualified_name"],
            "purpose": table.get("business_purpose"),
            "row_count": table.get("row_count"),
            "aliases": table.get("nl_aliases"),
            "columns": []
        }
        
        # Add key columns with semantic info
        for col in table.get("columns", []):
            col_info = {
                "name": col["name"],
                "type": col["data_type"],
                "semantic": col.get("semantic_role"),
                "aliases": col.get("nl_aliases"),
                "nullable": col.get("nullable")
            }
            
            # Add value intelligence for categoricals
            if col.get("top_values"):
                col_info["enum"] = [v[0] for v in col["top_values"][:5]]
            
            # Add range for numerics
            if col.get("min_value") is not None:
                col_info["range"] = [col.get("min_value"), col.get("max_value")]
            
            # Add governance flags
            if not col.get("expose_in_output", True):
                col_info["private"] = True
                
            table_info["columns"].append(col_info)
        
        schema_summary["tables"].append(table_info)
    
    # Add KPIs
    if enhanced_schema.get("kpis"):
        schema_summary["kpis"] = enhanced_schema["kpis"]
    
    # Combine documentation with schema summary
    schema_json = json.dumps(schema_summary, indent=2)
    
    return documentation_section + schema_json
