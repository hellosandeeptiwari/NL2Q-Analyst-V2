#!/usr/bin/env python3
"""
Test OpenAI Vector Matcher with sample schema data
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.agents.openai_vector_matcher import OpenAIVectorMatcher, SchemaItem

def test_vector_matcher():
    """Test the OpenAI vector matcher with sample data"""
    print("🧪 Testing OpenAI Vector Matcher\n")
    
    # Initialize matcher
    matcher = OpenAIVectorMatcher()
    
    # Sample schema items
    sample_tables = [
        SchemaItem(
            name="Final_NBA_Output_python_20250519",
            type="table",
            description="NBA basketball analytics output table with Python processed data from May 2025"
        ),
        SchemaItem(
            name="NBA_Player_Stats_2024",
            type="table", 
            description="NBA player statistics and performance data for 2024 season"
        ),
        SchemaItem(
            name="Customer_Orders_2024",
            type="table",
            description="Customer order transactions and purchase data for 2024"
        )
    ]
    
    sample_columns = [
        SchemaItem(
            name="player_name",
            type="column",
            table_name="NBA_Player_Stats_2024",
            data_type="VARCHAR",
            description="NBA player name identifier"
        ),
        SchemaItem(
            name="points_per_game",
            type="column", 
            table_name="NBA_Player_Stats_2024",
            data_type="FLOAT",
            description="Average points scored per game by player"
        )
    ]
    
    # Test adding schema items
    print("📊 Adding sample schema items...")
    for item in sample_tables + sample_columns:
        matcher.add_schema_item(item)
    
    print(f"✅ Added {len(sample_tables)} tables and {len(sample_columns)} columns\n")
    
    # Test building embeddings
    print("🔄 Building embeddings...")
    success = matcher.build_embeddings()
    
    if success:
        print("✅ Embeddings built successfully!\n")
        
        # Test similarity search
        test_queries = [
            "NBA basketball player data",
            "Customer purchase information", 
            "Player scoring statistics",
            "Points and game stats"
        ]
        
        print("🔍 Testing similarity search:")
        for query in test_queries:
            print(f"\n📋 Query: '{query}'")
            
            # Find similar tables
            similar_tables = matcher.find_similar_tables(query, top_k=2)
            if similar_tables:
                print("  Similar tables:")
                for table, score in similar_tables:
                    print(f"    • {table} (similarity: {score:.3f})")
            
            # Find similar columns  
            similar_columns = matcher.find_similar_columns(query, top_k=2)
            if similar_columns:
                print("  Similar columns:")
                for column, score in similar_columns:
                    print(f"    • {column} (similarity: {score:.3f})")
        
        # Test caching
        print(f"\n💾 Testing cache persistence...")
        cache_file = matcher.embedding_cache_file
        if os.path.exists(cache_file):
            file_size = os.path.getsize(cache_file)
            print(f"✅ Cache file created: {cache_file} ({file_size} bytes)")
        else:
            print("⚠️ Cache file not found")
            
    else:
        print("❌ Failed to build embeddings")

if __name__ == "__main__":
    try:
        test_vector_matcher()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
