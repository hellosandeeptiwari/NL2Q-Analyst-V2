"""
Performance test for 166 tables with ALL columns preserved
Tests optimizations without data loss
"""

import time
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def test_full_schema_performance():
    """Test embedding performance with all columns preserved"""
    print("🚀 Testing Full Schema Performance (No Column Reduction)")
    print("=" * 60)
    
    from backend.db.engine import get_adapter
    from backend.agents.openai_vector_matcher import OpenAIVectorMatcher
    
    adapter = get_adapter()
    vector_matcher = OpenAIVectorMatcher()
    
    start_time = time.time()
    
    # Test with cache first
    print("📋 Step 1: Check existing cache")
    cache_exists = vector_matcher._load_cached_embeddings()
    
    if cache_exists:
        print(f"✅ Cache found:")
        print(f"   📊 Tables: {len(vector_matcher.table_embeddings)}")
        print(f"   📊 Columns: {len(vector_matcher.column_embeddings)}")
        print(f"   📊 Total items: {len(vector_matcher.table_embeddings) + len(vector_matcher.column_embeddings)}")
    else:
        print("⚠️ No cache found")
    
    # Performance test with sample queries
    print(f"\n📋 Step 2: Query Performance Test")
    test_queries = [
        "NBA player statistics and performance data",
        "team scoring averages and win rates", 
        "player shooting percentages and rebounds",
        "game results and final scores",
        "season statistics and rankings"
    ]
    
    query_times = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: '{query}'")
        query_start = time.time()
        
        try:
            if cache_exists:
                # Test search performance
                search_results = vector_matcher.hybrid_search(query, top_k=5)
                
                query_time = time.time() - query_start
                query_times.append(query_time)
                
                print(f"   ⚡ Search: {query_time:.3f}s")
                print(f"   📊 Results: {len(search_results)}")
                
                # Show top matches
                for j, result in enumerate(search_results[:3]):
                    name = result.get('name', 'Unknown')
                    score = result.get('similarity', 0)
                    item_type = result.get('type', 'unknown')
                    print(f"   🎯 {j+1}. {name} ({item_type}) - score: {score:.3f}")
            else:
                print("   ⚠️ Skipping query test - no embeddings loaded")
                
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
    
    # Performance summary
    if query_times:
        avg_query_time = sum(query_times) / len(query_times)
        max_query_time = max(query_times)
        min_query_time = min(query_times)
        
        print(f"\n📊 Performance Summary:")
        print(f"   ⚡ Average query time: {avg_query_time:.3f}s")
        print(f"   ⚡ Fastest query: {min_query_time:.3f}s")
        print(f"   ⚡ Slowest query: {max_query_time:.3f}s")
        
        if avg_query_time < 1.0:
            print("   ✅ Excellent performance!")
        elif avg_query_time < 3.0:
            print("   ✅ Good performance")
        else:
            print("   ⚠️ Consider further optimization")
    
    total_time = time.time() - start_time
    print(f"\n✅ Test completed in {total_time:.2f} seconds")
    
    return cache_exists

def test_embedding_process():
    """Test the actual embedding process if no cache exists"""
    print("\n🔄 Testing Embedding Process (Full Schema)")
    
    from backend.db.engine import get_adapter
    from backend.agents.openai_vector_matcher import OpenAIVectorMatcher
    
    adapter = get_adapter()
    vector_matcher = OpenAIVectorMatcher()
    
    # Clear cache to test fresh embedding
    cache_files = [
        "backend/storage/schema_embeddings.pkl",
        "backend/storage/schema_metadata.json"
    ]
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            print(f"🗑️ Removing {cache_file} for fresh test")
            os.remove(cache_file)
    
    print("🚀 Starting fresh embedding process...")
    start_time = time.time()
    
    try:
        # Use optimized settings but keep all columns
        vector_matcher.initialize_from_database(
            adapter, 
            force_rebuild=True,
            max_tables=None,  # Process all tables
            important_tables=['NBA_FINAL_OUTPUT_PYTHON_DF']  # Prioritize this one
        )
        
        embedding_time = time.time() - start_time
        
        print(f"✅ Embedding completed!")
        print(f"   ⏱️ Total time: {embedding_time:.2f} seconds")
        print(f"   📊 Tables: {len(vector_matcher.table_embeddings)}")
        print(f"   📊 Columns: {len(vector_matcher.column_embeddings)}")
        print(f"   📊 Total items: {len(vector_matcher.table_embeddings) + len(vector_matcher.column_embeddings)}")
        
        # Calculate rate
        total_items = len(vector_matcher.table_embeddings) + len(vector_matcher.column_embeddings)
        items_per_second = total_items / embedding_time if embedding_time > 0 else 0
        
        print(f"   🚀 Processing rate: {items_per_second:.1f} items/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Full Schema Performance Test")
    print("Preserving ALL columns for complete data coverage")
    print("=" * 60)
    
    try:
        # Test performance with existing cache
        cache_exists = test_full_schema_performance()
        
        # If no cache, test the embedding process
        if not cache_exists:
            print("\n" + "="*60)
            user_input = input("No cache found. Test fresh embedding process? (y/n): ")
            if user_input.lower() in ['y', 'yes']:
                test_embedding_process()
            else:
                print("Skipping embedding test.")
        
        print(f"\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
