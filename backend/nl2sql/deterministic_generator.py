"""
Deterministic NL→SQL Generator with Column-First Approach
Configuration-driven, no hardcoded values
"""

import os
import openai
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import re
import json
from fuzzywuzzy import fuzz
from .guardrails import GuardrailConfig, sanitize_sql

@dataclass
class ScoringConfig:
    """Configuration for column scoring algorithm"""
    name_weight: float = 0.65
    type_weight: float = 0.30
    boost_weight: float = 0.05
    
    # Type compatibility scores
    high_compatibility: float = 0.9
    medium_compatibility: float = 0.7
    low_compatibility: float = 0.5
    default_compatibility: float = 0.3
    
    # Selection thresholds
    primary_threshold: float = 0.52
    fallback_threshold: float = 0.48
    min_columns_for_fallback: int = 3
    max_columns_selected: int = 12

@dataclass
class PatternConfig:
    """Configuration for pattern matching"""
    identifier_patterns: List[str] = field(default_factory=lambda: ['_id', 'id', 'code', 'key', 'identifier'])
    date_patterns: List[str] = field(default_factory=lambda: ['date', 'timestamp', 'time', 'datetime'])
    numeric_patterns: List[str] = field(default_factory=lambda: ['int', 'float', 'decimal', 'number', 'numeric', 'bigint'])
    text_patterns: List[str] = field(default_factory=lambda: ['varchar', 'text', 'string', 'char', 'nvarchar'])
    
    # Query intent keywords
    identifier_keywords: List[str] = field(default_factory=lambda: ['id', 'identifier', 'code', 'key', 'reference'])
    date_keywords: List[str] = field(default_factory=lambda: ['date', 'time', 'when', 'period', 'year', 'month', 'day', 'week'])
    numeric_keywords: List[str] = field(default_factory=lambda: ['count', 'sum', 'average', 'total', 'amount', 'value', 'price', 'cost', 'number', 'quantity'])
    text_keywords: List[str] = field(default_factory=lambda: ['name', 'description', 'title', 'label', 'text', 'comment'])

@dataclass
class JoinConfig:
    """Configuration for join planning"""
    default_join_type: str = "LEFT"
    require_exact_key_match: bool = True
    allow_fuzzy_key_matching: bool = False
    fuzzy_key_threshold: float = 0.85
    
    # Keywords that suggest row-level alignment is needed
    alignment_keywords: List[str] = field(default_factory=lambda: [
        'for each', 'per', 'by', 'breakdown', 'along with', 'together', 
        'combined', 'correlated', 'matching', 'paired'
    ])

@dataclass
class ColumnMatch:
    table_name: str
    column_name: str
    data_type: str
    score: float
    name_score: float
    type_score: float
    boost_score: float

@dataclass
class JoinPlan:
    required: bool
    result_grain: str
    steps: List[Dict[str, Any]]
    warnings: List[str]

@dataclass
class SQLPlan:
    status: str
    columns_selected: List[Dict[str, Any]]
    tables_selected: List[Dict[str, Any]]
    join: JoinPlan
    filters: Dict[str, Any]
    limits: Dict[str, int]
    confidence_overall: float
    diagnostics: Optional[Dict[str, Any]] = None
    unmatched_intents: Optional[List[str]] = None
    suggested_alternates: Optional[List[Dict[str, Any]]] = None

@dataclass
class GeneratedSQL:
    sql: str
    plan: SQLPlan
    rationale: str
    added_limit: bool
    suggestions: List[str]
    confidence_score: Optional[float] = None

class DeterministicNL2SQL:
    """Deterministic NL→SQL planner/generator with configurable, no-hardcode approach"""
    
    def __init__(self, scoring_config: Optional[ScoringConfig] = None, 
                 pattern_config: Optional[PatternConfig] = None,
                 join_config: Optional[JoinConfig] = None):
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.scoring_config = scoring_config or ScoringConfig()
        self.pattern_config = pattern_config or PatternConfig()
        self.join_config = join_config or JoinConfig()
        
    def normalize_tokens(self, text: str) -> List[str]:
        """Normalize tokens: case-insensitive, split snake/camel, singularize"""
        # Convert to lowercase and split on various delimiters
        tokens = re.split(r'[_\s\-\.]+|(?=[A-Z])', text.lower())
        # Simple singularization (remove trailing 's' in common cases)
        normalized = []
        for token in tokens:
            if token and len(token) > 1:
                if token.endswith('s') and not token.endswith('ss'):
                    normalized.append(token[:-1])
                normalized.append(token)
        return [t for t in normalized if t]
    
    def calculate_column_score(self, query_tokens: List[str], table_name: str, 
                             column_name: str, data_type: str, query_text: str) -> ColumnMatch:
        """Score columns using configurable weights and patterns"""
        
        # Normalize column and table names
        col_tokens = self.normalize_tokens(column_name)
        table_tokens = self.normalize_tokens(table_name)
        
        # S_name: token/fuzzy match between query and {table_name, column_name}
        name_scores = []
        for q_token in query_tokens:
            # Direct token matches
            col_matches = [fuzz.ratio(q_token, c_token) / 100.0 for c_token in col_tokens]
            table_matches = [fuzz.ratio(q_token, t_token) / 100.0 for t_token in table_tokens]
            
            if col_matches:
                name_scores.append(max(col_matches))
            if table_matches:
                name_scores.append(max(table_matches) * 0.7)  # Table matches slightly lower weight
        
        s_name = max(name_scores) if name_scores else 0.0
        
        # S_type: compatibility scoring using configurable patterns
        s_type = self.calculate_type_compatibility(query_text, data_type, column_name)
        
        # Configurable boost if table tokens occur in query
        table_boost = 0.0
        for t_token in table_tokens:
            if t_token in query_text.lower():
                table_boost = self.scoring_config.boost_weight
                break
        
        # Final score using configurable weights
        final_score = (self.scoring_config.name_weight * s_name + 
                      self.scoring_config.type_weight * s_type + 
                      table_boost)
        
        return ColumnMatch(
            table_name=table_name,
            column_name=column_name,
            data_type=data_type,
            score=final_score,
            name_score=s_name,
            type_score=s_type,
            boost_score=table_boost
        )
    
    def calculate_type_compatibility(self, query_text: str, data_type: str, column_name: str) -> float:
        """Calculate type compatibility score using configurable patterns"""
        query_lower = query_text.lower()
        col_lower = column_name.lower()
        type_lower = data_type.lower()
        
        # Identifier patterns
        if any(pattern in col_lower for pattern in self.pattern_config.identifier_patterns):
            if any(word in query_lower for word in self.pattern_config.identifier_keywords):
                return self.scoring_config.high_compatibility
        
        # Time/date patterns  
        if any(t in type_lower for t in self.pattern_config.date_patterns):
            if any(word in query_lower for word in self.pattern_config.date_keywords):
                return self.scoring_config.high_compatibility
        
        # Numeric/measure patterns
        if any(t in type_lower for t in self.pattern_config.numeric_patterns):
            if any(word in query_lower for word in self.pattern_config.numeric_keywords):
                return self.scoring_config.medium_compatibility
        
        # Text patterns
        if any(t in type_lower for t in self.pattern_config.text_patterns):
            if any(word in query_lower for word in self.pattern_config.text_keywords):
                return self.scoring_config.low_compatibility
        
        return self.scoring_config.default_compatibility
    
    def select_columns(self, natural_language: str, catalog: Dict[str, Any]) -> Tuple[List[ColumnMatch], List[str]]:
        """Column-first selection with configurable thresholds"""
        
        query_tokens = self.normalize_tokens(natural_language)
        all_matches = []
        
        # Score all columns
        tables = catalog.get('tables', [])
        columns = catalog.get('columns', [])
        
        for col in columns:
            match = self.calculate_column_score(
                query_tokens, 
                col['table_name'], 
                col['column_name'], 
                col['data_type'],
                natural_language
            )
            all_matches.append(match)
        
        # Sort by score and keep top-K (configurable)
        all_matches.sort(key=lambda x: x.score, reverse=True)
        
        # Use configurable thresholds
        threshold = self.scoring_config.primary_threshold
        filtered_matches = [m for m in all_matches if m.score >= threshold]
        
        # Fallback threshold if too few matches
        if len(filtered_matches) < self.scoring_config.min_columns_for_fallback:
            threshold = self.scoring_config.fallback_threshold
            filtered_matches = [m for m in all_matches if m.score >= threshold]
        
        # Keep top configurable number
        selected_matches = filtered_matches[:self.scoring_config.max_columns_selected]
        
        # Track unmatched intents
        unmatched = []
        if not selected_matches:
            unmatched.append(f"No columns found matching query concepts (threshold: {threshold})")
        
        return selected_matches, unmatched
    
    def determine_join_requirement(self, selected_columns: List[ColumnMatch], natural_language: str) -> JoinPlan:
        """Determine if joins are needed based on configurable patterns"""
        
        # Group columns by table
        tables_used = {}
        for col in selected_columns:
            if col.table_name not in tables_used:
                tables_used[col.table_name] = []
            tables_used[col.table_name].append(col)
        
        if len(tables_used) <= 1:
            return JoinPlan(
                required=False,
                result_grain="single-table",
                steps=[],
                warnings=[]
            )
        
        # Check if query requires same-row alignment using configurable keywords
        requires_alignment = any(keyword in natural_language.lower() 
                               for keyword in self.join_config.alignment_keywords)
        
        if not requires_alignment:
            return JoinPlan(
                required=False,
                result_grain="per-table",
                steps=[],
                warnings=["Multiple tables identified but no row-level alignment required"]
            )
        
        # Plan joins if alignment is required
        join_steps = []
        warnings = []
        
        table_names = list(tables_used.keys())
        for i in range(len(table_names) - 1):
            left_table = table_names[i]
            right_table = table_names[i + 1]
            
            # Look for join keys
            join_keys = self.find_join_keys(left_table, right_table, catalog)
            
            if not join_keys:
                warnings.append(f"No valid join keys found between {left_table} and {right_table}")
                continue
            
            join_steps.append({
                "left_table": left_table,
                "right_table": right_table,
                "join_type": self.join_config.default_join_type,
                "on": join_keys
            })
        
        return JoinPlan(
            required=True,
            result_grain="joined-tables",
            steps=join_steps,
            warnings=warnings
        )
    
    def find_join_keys(self, left_table: str, right_table: str, catalog: Dict[str, Any]) -> List[Dict[str, str]]:
        """Find valid join keys using configurable matching rules"""
        
        # Get columns for both tables
        left_cols = [col for col in catalog.get('columns', []) if col['table_name'] == left_table]
        right_cols = [col for col in catalog.get('columns', []) if col['table_name'] == right_table]
        
        join_keys = []
        
        if self.join_config.require_exact_key_match:
            # Exact name matches with identifier patterns
            for left_col in left_cols:
                for right_col in right_cols:
                    # Exact name match
                    if left_col['column_name'].lower() == right_col['column_name'].lower():
                        # Must have key pattern from config
                        if any(pattern in left_col['column_name'].lower() 
                              for pattern in self.pattern_config.identifier_patterns):
                            join_keys.append({
                                "left": f"{left_table}.{left_col['column_name']}",
                                "right": f"{right_table}.{right_col['column_name']}"
                            })
        
        if self.join_config.allow_fuzzy_key_matching and not join_keys:
            # Fuzzy matching for join keys
            for left_col in left_cols:
                for right_col in right_cols:
                    similarity = fuzz.ratio(left_col['column_name'].lower(), 
                                          right_col['column_name'].lower()) / 100.0
                    
                    if similarity >= self.join_config.fuzzy_key_threshold:
                        # Must have key pattern
                        if any(pattern in left_col['column_name'].lower() 
                              for pattern in self.pattern_config.identifier_patterns):
                            join_keys.append({
                                "left": f"{left_table}.{left_col['column_name']}",
                                "right": f"{right_table}.{right_col['column_name']}"
                            })
        
        # Check explicit foreign key relationships if available
        foreign_keys = catalog.get('keys', {}).get('foreign', [])
        for fk in foreign_keys:
            if ((fk['from_table'] == left_table and fk['to_table'] == right_table) or
                (fk['from_table'] == right_table and fk['to_table'] == left_table)):
                join_keys.append({
                    "left": f"{fk['from_table']}.{fk['from_column']}",
                    "right": f"{fk['to_table']}.{fk['to_column']}"
                })
        
        return join_keys
    
    def validate_plan(self, plan: SQLPlan, catalog: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Hard validation gate - must pass before SQL generation"""
        
        errors = []
        
        # Validate all tables exist
        available_tables = [t['name'] for t in catalog.get('tables', [])]
        for table_info in plan.tables_selected:
            if table_info['table'] not in available_tables:
                errors.append(f"Table '{table_info['table']}' not found in catalog")
        
        # Validate all columns exist
        available_columns = {(col['table_name'], col['column_name']) for col in catalog.get('columns', [])}
        for col_info in plan.columns_selected:
            if (col_info['table'], col_info['column']) not in available_columns:
                errors.append(f"Column '{col_info['table']}.{col_info['column']}' not found in catalog")
        
        # Validate join keys exist on both sides
        if plan.join.required:
            for step in plan.join.steps:
                # Parse join keys
                for join_key in step.get('on', []):
                    left_parts = join_key['left'].split('.')
                    right_parts = join_key['right'].split('.')
                    
                    if len(left_parts) != 2 or len(right_parts) != 2:
                        errors.append(f"Invalid join key format: {join_key}")
                        continue
                    
                    left_table, left_col = left_parts
                    right_table, right_col = right_parts
                    
                    if (left_table, left_col) not in available_columns:
                        errors.append(f"Left join key '{left_table}.{left_col}' not found")
                    if (right_table, right_col) not in available_columns:
                        errors.append(f"Right join key '{right_table}.{right_col}' not found")
        
        return len(errors) == 0, errors
    
    def generate_sql_from_plan(self, plan: SQLPlan, natural_language: str, 
                              database_type: str = "snowflake") -> str:
        """Generate SQL from validated plan"""
        
        if not plan.columns_selected:
            return "SELECT 'No columns selected' as error"
        
        # Build SELECT clause
        select_parts = []
        for col in plan.columns_selected:
            select_parts.append(f'"{col["table"]}"."{col["column"]}"')
        
        # Build FROM clause
        if not plan.join.required:
            # Single table or per-table queries
            if len(plan.tables_selected) == 1:
                table_name = plan.tables_selected[0]['table']
                from_clause = f'FROM "{table_name}"'
            else:
                # Multiple tables without joins - use UNION or separate queries
                from_clause = f'FROM "{plan.tables_selected[0]["table"]}"'
        else:
            # Build joins
            base_table = plan.join.steps[0]['left_table']
            from_clause = f'FROM "{base_table}"'
            
            for step in plan.join.steps:
                join_type = step['join_type']
                right_table = step['right_table']
                join_conditions = []
                
                for join_key in step['on']:
                    join_conditions.append(f'{join_key["left"]} = {join_key["right"]}')
                
                from_clause += f' {join_type} JOIN "{right_table}" ON {" AND ".join(join_conditions)}'
        
        # Build WHERE clause (time filters, etc.)
        where_parts = []
        if plan.filters.get('time'):
            where_parts.append(plan.filters['time'])
        
        where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
        
        # Build complete SQL
        sql = f"SELECT {', '.join(select_parts)} {from_clause}{where_clause}"
        
        # Add LIMIT
        limit = plan.limits.get('preview_rows', 100)
        sql += f" LIMIT {limit}"
        
        return sql
    
    def generate_deterministic_sql(self, natural_language: str, catalog: Dict[str, Any], 
                                 constraints: GuardrailConfig) -> GeneratedSQL:
        """Main entry point for deterministic SQL generation"""
        
        try:
            # Step 1: Parse intent (simplified for now)
            # TODO: Add more sophisticated intent parsing
            
            # Step 2: Select columns using scoring
            selected_columns, unmatched_intents = self.select_columns(natural_language, catalog)
            
            if not selected_columns:
                # Return error response
                plan = SQLPlan(
                    status="error",
                    columns_selected=[],
                    tables_selected=[],
                    join=JoinPlan(False, "", [], []),
                    filters={},
                    limits={},
                    confidence_overall=0.0,
                    unmatched_intents=unmatched_intents,
                    suggested_alternates=[]
                )
                
                return GeneratedSQL(
                    sql="",
                    plan=plan,
                    rationale="No columns found matching query concepts",
                    added_limit=False,
                    suggestions=[],
                    confidence_score=0.0
                )
            
            # Step 3: Group by table and decide joins
            join_plan = self.determine_join_requirement(selected_columns, natural_language)
            
            # Build table selection
            tables_used = {}
            for col in selected_columns:
                if col.table_name not in tables_used:
                    tables_used[col.table_name] = []
                tables_used[col.table_name].append(col.column_name)
            
            tables_selected = [
                {"table": table, "columns": columns} 
                for table, columns in tables_used.items()
            ]
            
            # Step 4: Create plan
            plan = SQLPlan(
                status="ok",
                columns_selected=[
                    {
                        "table": col.table_name,
                        "column": col.column_name,
                        "data_type": col.data_type,
                        "score": col.score
                    } for col in selected_columns
                ],
                tables_selected=tables_selected,
                join=join_plan,
                filters={},
                limits={"preview_rows": constraints.default_limit},
                confidence_overall=sum(col.score for col in selected_columns) / len(selected_columns)
            )
            
            # Step 5: Validation gate
            is_valid, validation_errors = self.validate_plan(plan, catalog)
            
            if not is_valid:
                plan.status = "error"
                plan.unmatched_intents = validation_errors
                
                return GeneratedSQL(
                    sql="",
                    plan=plan,
                    rationale="Validation failed: " + "; ".join(validation_errors),
                    added_limit=False,
                    suggestions=[],
                    confidence_score=0.0
                )
            
            # Step 6: Generate SQL
            sql = self.generate_sql_from_plan(plan, natural_language)
            
            # Apply guardrails
            safe_sql, added_limit = sanitize_sql(sql, constraints)
            
            return GeneratedSQL(
                sql=safe_sql,
                plan=plan,
                rationale="Generated using deterministic column-first approach",
                added_limit=added_limit,
                suggestions=[],
                confidence_score=plan.confidence_overall
            )
            
        except Exception as e:
            error_plan = SQLPlan(
                status="error",
                columns_selected=[],
                tables_selected=[],
                join=JoinPlan(False, "", [], []),
                filters={},
                limits={},
                confidence_overall=0.0,
                unmatched_intents=[f"Generation error: {str(e)}"]
            )
            
            return GeneratedSQL(
                sql="SELECT 'Error in SQL generation' as error",
                plan=error_plan,
                rationale=f"Error in deterministic generation: {str(e)}",
                added_limit=False,
                suggestions=[],
                confidence_score=0.0
            )

# Main function for backward compatibility with configuration support
def generate_deterministic_sql(natural_language: str, catalog: Dict[str, Any], 
                             constraints: GuardrailConfig,
                             config_path: Optional[str] = None) -> GeneratedSQL:
    """Generate SQL using deterministic column-first approach with optional configuration"""
    
    # Load configuration from file or environment if provided
    scoring_config, pattern_config, join_config = load_configurations(config_path)
    
    generator = DeterministicNL2SQL(scoring_config, pattern_config, join_config)
    return generator.generate_deterministic_sql(natural_language, catalog, constraints)

def load_configurations(config_path: Optional[str] = None) -> Tuple[ScoringConfig, PatternConfig, JoinConfig]:
    """Load configurations from file, environment, or use defaults"""
    
    scoring_config = ScoringConfig()
    pattern_config = PatternConfig()
    join_config = JoinConfig()
    
    # Load from JSON file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update scoring config from file
            if 'scoring' in config_data:
                scoring_data = config_data['scoring']
                for key, value in scoring_data.items():
                    if hasattr(scoring_config, key):
                        setattr(scoring_config, key, value)
            
            # Update pattern config from file
            if 'patterns' in config_data:
                pattern_data = config_data['patterns']
                for key, value in pattern_data.items():
                    if hasattr(pattern_config, key):
                        setattr(pattern_config, key, value)
            
            # Update join config from file
            if 'joins' in config_data:
                join_data = config_data['joins']
                for key, value in join_data.items():
                    if hasattr(join_config, key):
                        setattr(join_config, key, value)
                        
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
    
    # Override with environment variables if set
    scoring_config = load_scoring_from_env(scoring_config)
    pattern_config = load_patterns_from_env(pattern_config)
    join_config = load_joins_from_env(join_config)
    
    return scoring_config, pattern_config, join_config

def load_scoring_from_env(config: ScoringConfig) -> ScoringConfig:
    """Load scoring configuration from environment variables"""
    
    env_mappings = {
        'DETERMINISTIC_NAME_WEIGHT': 'name_weight',
        'DETERMINISTIC_TYPE_WEIGHT': 'type_weight',
        'DETERMINISTIC_BOOST_WEIGHT': 'boost_weight',
        'DETERMINISTIC_HIGH_COMPATIBILITY': 'high_compatibility',
        'DETERMINISTIC_MEDIUM_COMPATIBILITY': 'medium_compatibility',
        'DETERMINISTIC_LOW_COMPATIBILITY': 'low_compatibility',
        'DETERMINISTIC_DEFAULT_COMPATIBILITY': 'default_compatibility',
        'DETERMINISTIC_PRIMARY_THRESHOLD': 'primary_threshold',
        'DETERMINISTIC_FALLBACK_THRESHOLD': 'fallback_threshold',
        'DETERMINISTIC_MIN_COLUMNS_FALLBACK': 'min_columns_for_fallback',
        'DETERMINISTIC_MAX_COLUMNS': 'max_columns_selected'
    }
    
    for env_var, attr_name in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value:
            try:
                if attr_name in ['min_columns_for_fallback', 'max_columns_selected']:
                    setattr(config, attr_name, int(env_value))
                else:
                    setattr(config, attr_name, float(env_value))
            except ValueError:
                print(f"Warning: Invalid value for {env_var}: {env_value}")
    
    return config

def load_patterns_from_env(config: PatternConfig) -> PatternConfig:
    """Load pattern configuration from environment variables"""
    
    list_mappings = {
        'DETERMINISTIC_ID_PATTERNS': 'identifier_patterns',
        'DETERMINISTIC_DATE_PATTERNS': 'date_patterns',
        'DETERMINISTIC_NUMERIC_PATTERNS': 'numeric_patterns',
        'DETERMINISTIC_TEXT_PATTERNS': 'text_patterns',
        'DETERMINISTIC_ID_KEYWORDS': 'identifier_keywords',
        'DETERMINISTIC_DATE_KEYWORDS': 'date_keywords',
        'DETERMINISTIC_NUMERIC_KEYWORDS': 'numeric_keywords',
        'DETERMINISTIC_TEXT_KEYWORDS': 'text_keywords'
    }
    
    for env_var, attr_name in list_mappings.items():
        env_value = os.getenv(env_var)
        if env_value:
            try:
                # Split by comma and strip whitespace
                value_list = [item.strip() for item in env_value.split(',')]
                setattr(config, attr_name, value_list)
            except Exception as e:
                print(f"Warning: Could not parse {env_var}: {e}")
    
    return config

def load_joins_from_env(config: JoinConfig) -> JoinConfig:
    """Load join configuration from environment variables"""
    
    env_mappings = {
        'DETERMINISTIC_DEFAULT_JOIN_TYPE': ('default_join_type', str),
        'DETERMINISTIC_REQUIRE_EXACT_KEYS': ('require_exact_key_match', lambda x: x.lower() == 'true'),
        'DETERMINISTIC_ALLOW_FUZZY_KEYS': ('allow_fuzzy_key_matching', lambda x: x.lower() == 'true'),
        'DETERMINISTIC_FUZZY_KEY_THRESHOLD': ('fuzzy_key_threshold', float)
    }
    
    for env_var, (attr_name, converter) in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value:
            try:
                setattr(config, attr_name, converter(env_value))
            except (ValueError, AttributeError) as e:
                print(f"Warning: Invalid value for {env_var}: {env_value} ({e})")
    
    # Handle alignment keywords list
    alignment_keywords_env = os.getenv('DETERMINISTIC_ALIGNMENT_KEYWORDS')
    if alignment_keywords_env:
        try:
            config.alignment_keywords = [kw.strip() for kw in alignment_keywords_env.split(',')]
        except Exception as e:
            print(f"Warning: Could not parse DETERMINISTIC_ALIGNMENT_KEYWORDS: {e}")
    
    return config

def create_sample_config_file(config_path: str) -> None:
    """Create a sample configuration file showing all available options"""
    
    sample_config = {
        "scoring": {
            "name_weight": 0.65,
            "type_weight": 0.30,
            "boost_weight": 0.05,
            "high_compatibility": 0.9,
            "medium_compatibility": 0.7,
            "low_compatibility": 0.5,
            "default_compatibility": 0.3,
            "primary_threshold": 0.52,
            "fallback_threshold": 0.48,
            "min_columns_for_fallback": 3,
            "max_columns_selected": 12
        },
        "patterns": {
            "identifier_patterns": ["_id", "id", "code", "key", "identifier"],
            "date_patterns": ["date", "timestamp", "time", "datetime"],
            "numeric_patterns": ["int", "float", "decimal", "number", "numeric", "bigint"],
            "text_patterns": ["varchar", "text", "string", "char", "nvarchar"],
            "identifier_keywords": ["id", "identifier", "code", "key", "reference"],
            "date_keywords": ["date", "time", "when", "period", "year", "month", "day", "week"],
            "numeric_keywords": ["count", "sum", "average", "total", "amount", "value", "price", "cost", "number", "quantity"],
            "text_keywords": ["name", "description", "title", "label", "text", "comment"]
        },
        "joins": {
            "default_join_type": "LEFT",
            "require_exact_key_match": True,
            "allow_fuzzy_key_matching": False,
            "fuzzy_key_threshold": 0.85,
            "alignment_keywords": ["for each", "per", "by", "breakdown", "along with", "together", "combined", "correlated", "matching", "paired"]
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"Sample configuration file created at: {config_path}")
    print("You can customize these values to tune the deterministic SQL generation behavior.")
