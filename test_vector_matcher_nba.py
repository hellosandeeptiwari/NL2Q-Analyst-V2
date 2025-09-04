#!/usr/bin/env python3
"""
Test Vector Matcher NBA Table Discovery
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

async def test_vector_matcher_nba():
    """Test if vector matcher can find NBA table"""
    
    print("🔍 Testing Vector Matcher NBA Discovery...")
    
    try:
        from backend.agents.openai_vector_matcher import OpenAIVectorMatcher
        from backend.db.enhanced_schema import get_enhanced_schema_cache
        
        # Create vector matcher
        vector_matcher = OpenAIVectorMatcher()
        
        # Test queries
        test_queries = [
            "read table final nba output python",
            "nba data similarity output",
            "nba phase2 table",
            "similarity output final python",
            "NBA_PHASE2_SEP2024_SIMILARITY_OUTPUT_FINAL_PYTHON",
            "nba",
            "basketball"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            
            try:
                # Find similar tables
                matches = vector_matcher.find_similar_tables(query, top_k=5)
                
                print(f"   Found {len(matches)} matches:")
                for i, match in enumerate(matches[:3]):
                    table_name = match.get("table_name", "unknown")
                    score = match.get("similarity_score", 0)
                    print(f"   {i+1}. {table_name} (score: {score:.3f})")
                    
                # Check if NBA table is in any results
                nba_found = any("NBA" in match.get("table_name", "").upper() for match in matches)
                print(f"   NBA table found: {'✅' if nba_found else '❌'}")
                
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
        
        # Also check the enhanced schema cache directly
        print(f"\n📋 Checking Enhanced Schema Cache...")
        schema_cache = get_enhanced_schema_cache()
        
        nba_tables = [name for name in schema_cache.keys() if "NBA" in name.upper()]
        print(f"   NBA tables in cache: {len(nba_tables)}")
        for table in nba_tables:
            print(f"   • {table}")
            
        if nba_tables:
            # Show details of first NBA table
            first_nba = nba_tables[0]
            table_info = schema_cache[first_nba]
            print(f"\n📊 NBA Table Details:")
            print(f"   Name: {first_nba}")
            print(f"   Columns: {len(table_info.get('columns', {}))}")
            print(f"   Row count: {table_info.get('row_count', 'unknown')}")
            
            # Show first few columns
            columns = list(table_info.get('columns', {}).keys())
            print(f"   Sample columns: {columns[:5]}")
        
        return nba_tables
        
    except Exception as e:
        print(f"❌ Vector matcher test failed: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    nba_tables = asyncio.run(test_vector_matcher_nba())
    
    if nba_tables:
        print(f"\n✅ NBA tables available in system")
    else:
        print(f"\n❌ No NBA tables found - need to refresh schema cache")
