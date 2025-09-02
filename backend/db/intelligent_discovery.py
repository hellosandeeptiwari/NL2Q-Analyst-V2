"""
Intelligent Table Discovery for NBA Database
Finds relevant tables using fuzzy matching and semantic similarity
"""
import re
from difflib import SequenceMatcher

def discover_nba_tables(adapter, query_text=""):
    """
    Intelligently discover NBA-related tables in the database
    
    Args:
        adapter: Database adapter
        query_text: The natural language query to help guide table selection
    
    Returns:
        List of relevant table names
    """
    try:
        # Query database for all available tables
        discovery_query = """
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
        ORDER BY TABLE_NAME
        """
        
        result = adapter.run(discovery_query)
        all_tables = [row[0] for row in result.rows]
        
        print(f"🔍 Found {len(all_tables)} total tables in database")
        
        # Find NBA-related tables using multiple strategies
        nba_tables = []
        
        # Strategy 1: Direct NBA keyword matching
        nba_keyword_tables = [t for t in all_tables if 'NBA' in t.upper()]
        nba_tables.extend(nba_keyword_tables)
        
        # Strategy 2: Basketball/sports related keywords
        sports_keywords = ['BASKETBALL', 'PLAYER', 'TEAM', 'GAME', 'SEASON', 'SPORT', 'SCORE', 'MATCH']
        for keyword in sports_keywords:
            keyword_tables = [t for t in all_tables if keyword in t.upper()]
            nba_tables.extend(keyword_tables)
        
        # Strategy 3: Query-specific table matching
        if query_text:
            query_upper = query_text.upper()
            # Extract potential table names from the query
            potential_tables = re.findall(r'\b[A-Z_]+\b', query_upper)
            for potential in potential_tables:
                # Find similar table names using fuzzy matching
                for table in all_tables:
                    similarity = SequenceMatcher(None, potential, table.upper()).ratio()
                    if similarity > 0.6:  # 60% similarity threshold
                        nba_tables.append(table)
                        print(f"🎯 Found similar table: {table} (similarity: {similarity:.2f}) for query term: {potential}")
        
        # Remove duplicates and limit results
        nba_tables = list(set(nba_tables))[:10]  # Limit to top 10 tables
        
        print(f"🏀 Discovered NBA/Sports tables: {nba_tables}")
        
        # If no NBA tables found, return some general tables for exploration
        if not nba_tables:
            print("⚠️ No NBA-specific tables found, using general table discovery")
            nba_tables = all_tables[:5]  # Use first 5 tables as fallback
        
        return nba_tables
        
    except Exception as e:
        print(f"Error in table discovery: {e}")
        return []

def build_intelligent_schema(adapter, query_text=""):
    """
    Build schema using intelligent table discovery
    
    Args:
        adapter: Database adapter  
        query_text: Natural language query to guide table selection
        
    Returns:
        Dict with schema information for relevant tables
    """
    # Discover relevant tables
    relevant_tables = discover_nba_tables(adapter, query_text)
    
    if not relevant_tables:
        print("❌ No relevant tables discovered")
        return {}
    
    # Build minimal schema for discovered tables
    from backend.db.minimal_schema import build_minimal_schema
    schema = build_minimal_schema(adapter, relevant_tables)
    
    print(f"✅ Built intelligent schema with {len(schema)} tables")
    return schema
