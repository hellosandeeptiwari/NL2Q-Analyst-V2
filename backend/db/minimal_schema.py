"""Minimal schema builder for efficient SQL generation"""

def build_minimal_schema(adapter, target_tables=None):
    """
    Build a minimal schema with only essential information:
    - Table names
    - Column names  
    - Column data types
    - Column descriptions (if available)
    
    Args:
        adapter: Database adapter
        target_tables: List of specific tables to include (optional)
    
    Returns:
        Dict with minimal schema information
    """
    minimal_schema = {}
    
    try:
        # Get basic table and column info from INFORMATION_SCHEMA
        query = """
        SELECT 
            TABLE_NAME,
            COLUMN_NAME,
            DATA_TYPE,
            COALESCE(COMMENT, '') as DESCRIPTION
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
        """
        
        # If specific tables requested, filter for those
        if target_tables:
            table_list = "', '".join(target_tables)
            query += f" AND TABLE_NAME IN ('{table_list}')"
            
        query += " ORDER BY TABLE_NAME, ORDINAL_POSITION LIMIT 50"  # Limit columns
        
        result = adapter.run(query)
        
        for row in result.rows:
            table_name = row[0]
            column_name = row[1] 
            data_type = row[2]
            description = row[3] if len(row) > 3 else ""
            
            if table_name not in minimal_schema:
                minimal_schema[table_name] = {}
                
            # Store minimal column info
            column_info = data_type
            if description:
                column_info += f" -- {description}"
                
            minimal_schema[table_name][column_name] = column_info
            
    except Exception as e:
        print(f"Warning: Could not build minimal schema: {e}")
        # Fallback to hardcoded minimal schema
        minimal_schema = {
            "FINAL_NBA_OUTPUT_PYTHON": {
                "id": "VARCHAR",
                "recommendation": "VARCHAR", 
                "provider": "VARCHAR",
                "message": "VARCHAR"
            }
        }
    
    return minimal_schema
