"""
Test script for large schema embedding optimization
Tests the new optimizations with 166 tables and 3000+ schema objects
"""

import time
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.db.engine import get_adapter
from backend.agents.openai_vector_matcher import OpenAIVectorMatcher
from backend.utils.smart_schema_manager import SmartSchemaManager

def test_large_schema_optimization():
    """Test optimized schema embedding for large databases"""
    print("🚀 Testing Large Schema Optimization")
    print("=" * 50)
    
    # Initialize components
    adapter = get_adapter()
    vector_matcher = OpenAIVectorMatcher()
    manager = SmartSchemaManager()
    
    start_time = time.time()
    
    # Test 1: Check if cache exists
    print("\n📋 Test 1: Cache Check")
    cache_exists = vector_matcher._load_cached_embeddings()
    if cache_exists:
        print(f"✅ Cache found - {len(vector_matcher.table_embeddings)} tables, {len(vector_matcher.column_embeddings)} columns")
    else:
        print("⚠️ No cache found - will need to build embeddings")
    
    # Test 2: Get table count and recommendations
    print("\n📋 Test 2: Schema Analysis")
    try:
        # Try to get table count
        result = adapter.run("SELECT COUNT(*) as table_count FROM INFORMATION_SCHEMA.TABLES")
        if result and result.rows:
            total_tables = result.rows[0][0] if result.rows[0] else 0
        else:
            # Fallback method
            result = adapter.run("SHOW TABLES")
            total_tables = len(result.rows) if result and result.rows else 0
            
        print(f"📊 Total tables found: {total_tables}")
        
        # Get recommendations
        estimated_items = total_tables * 21  # Estimate
        recommendations = manager.get_optimization_recommendations(total_tables, estimated_items)
        
        print("\n🎯 Optimization Recommendations:")
        for rec in recommendations:
            print(f"   {rec}")
            
        # Get recommended batch size
        batch_size = manager.get_recommended_batch_size(estimated_items)
        print(f"\n📦 Recommended batch size: {batch_size}")
        
        # Estimate processing time
        processing_time = manager.estimate_processing_time(estimated_items, batch_size)
        print(f"⏱️ Estimated processing time: {processing_time}")
        
    except Exception as e:
        print(f"❌ Error analyzing schema: {e}")
        return False
    
    # Test 3: Optimized initialization (if needed)
    if not cache_exists and total_tables > 0:
        print(f"\n📋 Test 3: Optimized Initialization")
        print(f"🔄 Building embeddings for {total_tables} tables...")
        
        # Define important tables for NBA analysis
        important_tables = [
            'NBA_FINAL_OUTPUT_PYTHON_DF',
            # Add more based on your specific schema
        ]
        
        init_start = time.time()
        
        try:
            if total_tables > 50:
                print("📊 Using large schema optimizations")
                vector_matcher.initialize_from_database(
                    adapter, 
                    force_rebuild=False,
                    max_tables=min(100, total_tables),  # Limit for testing
                    important_tables=important_tables
                )
            else:
                print("📊 Using standard initialization")
                vector_matcher.initialize_from_database(adapter, force_rebuild=False)
                
            init_time = time.time() - init_start
            print(f"✅ Initialization completed in {init_time:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Error during initialization: {e}")
            return False
    
    # Test 4: Query performance test
    print(f"\n📋 Test 4: Query Performance")
    test_queries = [
        "show me NBA player statistics",
        "find team performance data",
        "get player scoring information"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: '{query}'")
        query_start = time.time()
        
        try:
            # Test table search
            search_results = vector_matcher.hybrid_search(query, top_k=3)
            
            query_time = time.time() - query_start
            print(f"   ⚡ Search completed in {query_time:.3f} seconds")
            print(f"   📊 Found {len(search_results)} relevant items")
            
            # Show top result
            if search_results:
                top_result = search_results[0]
                print(f"   🎯 Top match: {top_result.get('name', 'Unknown')} (score: {top_result.get('similarity', 0):.3f})")
                
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
    
    total_time = time.time() - start_time
    print(f"\n✅ All tests completed in {total_time:.2f} seconds")
    
    return True

def test_memory_usage():
    """Test memory usage with large schemas"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"\n💾 Memory Usage Test")
    print(f"📊 Memory before: {memory_before:.1f} MB")
    
    # Run the main test
    test_large_schema_optimization()
    
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = memory_after - memory_before
    
    print(f"📊 Memory after: {memory_after:.1f} MB")
    print(f"📊 Memory used: {memory_used:.1f} MB")
    
    if memory_used > 500:  # More than 500MB
        print("⚠️ High memory usage detected - consider further optimization")
    else:
        print("✅ Memory usage within acceptable limits")

if __name__ == "__main__":
    print("🧪 Large Schema Embedding Test Suite")
    print("Testing optimizations for 166 tables with 3000+ schema objects")
    print("=" * 60)
    
    try:
        # Run basic test
        success = test_large_schema_optimization()
        
        if success:
            print("\n🎉 Basic test passed!")
            
            # Run memory test if psutil is available
            try:
                test_memory_usage()
            except ImportError:
                print("⚠️ psutil not available - skipping memory test")
                print("   Install with: pip install psutil")
        else:
            print("\n❌ Basic test failed!")
            
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
