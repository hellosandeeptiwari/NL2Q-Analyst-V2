#!/usr/bin/env python3
"""
Test fast schema retrieval performance
"""
import sys
import os
from pathlib import Path
import time

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from dotenv import load_dotenv
load_dotenv()

def test_fast_schema_retrieval():
    """Test the optimized schema retrieval performance"""
    print("🚀 Testing Fast Schema Retrieval Performance\n")
    
    try:
        from agents.schema_embedder import SchemaEmbedder
        from db.engine import get_adapter
        
        # Initialize optimized embedder
        embedder = SchemaEmbedder(
            api_key=os.getenv('OPENAI_API_KEY'),
            batch_size=20,
            max_workers=3  # Conservative for testing
        )
        
        adapter = get_adapter()
        
        # Test 1: Fresh extraction (no cache)
        print("🔄 Test 1: Fresh schema extraction...")
        start_time = time.time()
        
        schemas = embedder.extract_schema_from_db(adapter, max_workers=3, use_cache=False)
        
        fresh_time = time.time() - start_time
        print(f"✅ Fresh extraction: {len(schemas)} schemas in {fresh_time:.2f}s")
        print(f"   📊 Rate: {len(schemas)/fresh_time:.1f} tables/second")
        
        # Test 2: Cached extraction
        print(f"\n🔄 Test 2: Cached schema extraction...")
        start_time = time.time()
        
        cached_schemas = embedder.extract_schema_from_db(adapter, max_workers=3, use_cache=True)
        
        cached_time = time.time() - start_time
        print(f"✅ Cached extraction: {len(cached_schemas)} schemas in {cached_time:.2f}s")
        
        # Performance comparison
        print(f"\n📊 Performance Comparison:")
        print(f"   Fresh extraction: {fresh_time:.2f}s")
        print(f"   Cached extraction: {cached_time:.2f}s")
        if fresh_time > 0:
            speedup = fresh_time / cached_time if cached_time > 0 else float('inf')
            print(f"   Speedup: {speedup:.1f}x faster")
        
        # Test a subset for embedding
        print(f"\n🔄 Test 3: Quick embedding test (10 tables)...")
        subset_schemas = dict(list(schemas.items())[:10])
        
        start_time = time.time()
        embedded_subset = embedder.create_embeddings(subset_schemas)
        embedding_time = time.time() - start_time
        
        print(f"✅ Embedded {len(embedded_subset)} tables in {embedding_time:.2f}s")
        print(f"   📊 Embedding rate: {len(embedded_subset)/embedding_time:.1f} tables/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fast_schema_retrieval()
    if success:
        print("\n🎉 Fast schema retrieval test completed!")
        print("💡 Key optimizations:")
        print("   • Parallel schema extraction with threading")
        print("   • Smart caching (24-hour TTL)")
        print("   • Optimized SQL queries")
        print("   • Batch embedding processing")
        print("   • Fast column type detection")
    else:
        print("\n❌ Fast schema retrieval test failed")
