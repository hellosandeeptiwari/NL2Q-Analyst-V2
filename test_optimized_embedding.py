#!/usr/bin/env python3
"""
Test optimized schema embedding performance
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

def test_optimized_embedding():
    """Test the optimized embedding performance"""
    print("🚀 Testing Optimized Schema Embedding Performance\n")
    
    try:
        # Import optimized embedder
        from agents.schema_embedder import SchemaEmbedder
        from db.engine import get_adapter
        
        # Initialize with optimization settings
        embedder = SchemaEmbedder(
            api_key=os.getenv('OPENAI_API_KEY'),
            batch_size=20,  # Smaller batches for testing
            max_workers=2   # Conservative threading
        )
        
        # Get database adapter
        adapter = get_adapter()
        
        # Load existing cache if available
        cache_loaded = embedder.load_cache()
        if cache_loaded:
            print(f"📁 Loaded {len(embedder.schemas)} cached schemas")
        
        # Extract fresh schemas from database
        print("📋 Extracting schema from database...")
        start_time = time.time()
        
        schemas = embedder.extract_schema_from_db(adapter)
        extract_time = time.time() - start_time
        
        print(f"✅ Extracted {len(schemas)} schemas in {extract_time:.2f}s")
        
        # Test embedding process
        print(f"\n🔄 Testing optimized embedding process...")
        embedding_start = time.time()
        
        embedded_schemas = embedder.create_embeddings(schemas)
        
        embedding_time = time.time() - embedding_start
        
        print(f"\n📊 Performance Results:")
        print(f"   Schema extraction: {extract_time:.2f}s")
        print(f"   Embedding process: {embedding_time:.2f}s")
        print(f"   Total schemas: {len(embedded_schemas)}")
        
        # Show stats
        stats = embedder.stats
        print(f"\n📈 Detailed Stats:")
        print(f"   Total tables: {stats['total_tables']}")
        print(f"   Cached tables: {stats['cached_tables']}")
        print(f"   Newly embedded: {stats['embedded_tables']}")
        print(f"   Skipped tables: {stats['skipped_tables']}")
        
        if stats['embedded_tables'] > 0:
            rate = stats['embedded_tables'] / stats['embedding_time']
            print(f"   Embedding rate: {rate:.1f} tables/second")
        
        # Test similarity search
        if embedded_schemas:
            print(f"\n🔍 Testing similarity search...")
            test_queries = [
                "NBA basketball player statistics",
                "Customer transaction data",
                "Product inventory information"
            ]
            
            for query in test_queries:
                results = embedder.find_relevant_tables(query, top_k=3)
                print(f"   Query: '{query}'")
                for table, score in results[:3]:
                    print(f"     • {table} ({score:.3f})")
                print()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_optimized_embedding()
    if success:
        print("🎉 Optimized embedding test completed successfully!")
    else:
        print("❌ Optimized embedding test failed")
