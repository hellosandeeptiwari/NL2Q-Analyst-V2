"""
Schema Optimization Configuration for Large Databases
Handles 100+ tables with 3000+ schema objects efficiently
"""

class SchemaOptimizationConfig:
    """Configuration for optimizing large schema embeddings"""
    
    # Embedding batch settings
    EMBEDDING_BATCH_SIZE = 50  # OpenAI supports up to 2048
    EMBEDDING_TIMEOUT = 120.0  # Seconds
    EMBEDDING_MAX_RETRIES = 3
    
    # Schema filtering settings
    MAX_TABLES_AUTO = 100  # Auto-limit for very large schemas
    MAX_COLUMNS_PER_TABLE = None  # Keep all columns to preserve data completeness
    
    # Important table patterns (prioritized first)
    IMPORTANT_TABLE_PATTERNS = [
        'main', 'core', 'primary', 'fact', 'dim', 'dimension',
        'lookup', 'ref', 'reference', 'master', 'nba', 'final',
        'output', 'result', 'summary', 'agg', 'aggregate'
    ]
    
    # Table patterns to deprioritize (processed last)
    LOW_PRIORITY_PATTERNS = [
        'temp', 'tmp', 'backup', 'bak', 'test', 'debug',
        'log', 'audit', 'archive', 'old', 'deprecated'
    ]
    
    # Column patterns to prioritize
    IMPORTANT_COLUMN_PATTERNS = [
        'id', 'key', 'name', 'title', 'date', 'time',
        'amount', 'value', 'count', 'score', 'rate',
        'status', 'type', 'category', 'description'
    ]
    
    @classmethod
    def get_table_priority(cls, table_name: str) -> int:
        """
        Get priority score for a table (higher = more important)
        Returns: 0-100 priority score
        """
        name_lower = table_name.lower()
        score = 50  # Default score
        
        # Boost for important patterns
        for pattern in cls.IMPORTANT_TABLE_PATTERNS:
            if pattern in name_lower:
                score += 20
                break
        
        # Reduce for low priority patterns
        for pattern in cls.LOW_PRIORITY_PATTERNS:
            if pattern in name_lower:
                score -= 30
                break
        
        # Boost for specific NBA patterns
        if 'nba' in name_lower:
            score += 15
        if 'final' in name_lower and 'output' in name_lower:
            score += 10
            
        return max(0, min(100, score))
    
    @classmethod
    def filter_and_sort_tables(cls, tables: list, max_tables: int = None) -> list:
        """
        Filter and sort tables by importance
        """
        # Calculate priorities
        table_priorities = [(table, cls.get_table_priority(table)) for table in tables]
        
        # Sort by priority (highest first)
        table_priorities.sort(key=lambda x: x[1], reverse=True)
        
        # Extract sorted table names
        sorted_tables = [table for table, _ in table_priorities]
        
        # Apply limit if specified
        if max_tables and len(sorted_tables) > max_tables:
            sorted_tables = sorted_tables[:max_tables]
            
        return sorted_tables
    
    @classmethod
    def get_column_priority(cls, column_name: str, column_type: str) -> int:
        """
        Get priority score for a column (higher = more important)
        """
        name_lower = column_name.lower()
        type_lower = column_type.lower() if column_type else ""
        score = 50  # Default score
        
        # Boost for important column patterns
        for pattern in cls.IMPORTANT_COLUMN_PATTERNS:
            if pattern in name_lower:
                score += 15
                break
        
        # Boost for primary keys and foreign keys
        if 'id' in name_lower and ('key' in name_lower or name_lower.endswith('_id')):
            score += 25
        
        # Boost for common data types
        if any(t in type_lower for t in ['varchar', 'text', 'string']):
            score += 5
        elif any(t in type_lower for t in ['int', 'number', 'decimal', 'float']):
            score += 5
        elif any(t in type_lower for t in ['date', 'time', 'timestamp']):
            score += 10
            
        return max(0, min(100, score))

# Default configuration instance
schema_config = SchemaOptimizationConfig()
