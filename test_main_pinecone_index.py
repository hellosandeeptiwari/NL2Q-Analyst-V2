"""
Test the main Pinecone index and explore its contents
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import json

# Load environment variables
load_dotenv()

def test_main_pinecone_index():
    """Test the main Pinecone index configured in .env"""
    print("🔍 Testing main Pinecone index...")
    
    # Get configuration from .env
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    index_name = os.getenv('PINECONE_INDEX_NAME', 'nl2q-schema-index-3072')
    
    print(f"📊 Index Name: {index_name}")
    
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Connect to the specific index
        index = pc.Index(index_name)
        
        # Get index stats
        stats = index.describe_index_stats()
        print(f"✅ Connected successfully!")
        print(f"📈 Total Vectors: {stats.total_vector_count}")
        print(f"🏷️  Namespaces: {list(stats.namespaces.keys()) if stats.namespaces else ['default']}")
        
        # Try to fetch a few sample vectors to see the metadata structure
        print("\n🔍 Sample vector metadata:")
        try:
            # Query for some vectors (without a specific query vector, just to see structure)
            # We'll use a dummy vector for this purpose
            dummy_vector = [0.1] * 3072  # 3072 dimensions
            
            results = index.query(
                vector=dummy_vector,
                top_k=3,
                include_metadata=True
            )
            
            for i, match in enumerate(results.matches, 1):
                print(f"\n   Sample {i}:")
                print(f"   ID: {match.id}")
                print(f"   Score: {match.score:.4f}")
                if match.metadata:
                    print(f"   Metadata keys: {list(match.metadata.keys())}")
                    # Show some key metadata fields
                    for key in ['table_name', 'chunk_type', 'database', 'schema']:
                        if key in match.metadata:
                            print(f"   {key}: {match.metadata[key]}")
        
        except Exception as e:
            print(f"   ⚠️  Could not fetch sample data: {str(e)}")
    
    except Exception as e:
        print(f"❌ Error testing index: {str(e)}")

if __name__ == "__main__":
    test_main_pinecone_index()
