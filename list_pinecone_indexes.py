"""
List all available Pinecone indexes
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

def list_pinecone_indexes():
    """List all available Pinecone indexes"""
    print("🔍 Checking Pinecone configuration...")
    
    # Get Pinecone API key
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    if not pinecone_api_key:
        print("❌ PINECONE_API_KEY not found in .env file")
        return
    
    print(f"✅ Pinecone API key found: {pinecone_api_key[:10]}...")
    
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=pinecone_api_key)
        
        # List all indexes
        indexes = pc.list_indexes()
        
        print(f"\n📋 Available Pinecone Indexes ({len(indexes)} total):")
        print("=" * 60)
        
        if not indexes:
            print("No indexes found in your Pinecone account")
            return
            
        for i, index in enumerate(indexes, 1):
            print(f"{i}. Index Name: {index.name}")
            print(f"   Dimension: {index.dimension}")
            print(f"   Metric: {index.metric}")
            print(f"   Status: {index.status.ready}")
            print(f"   Host: {index.host}")
            print("-" * 40)
            
            # Try to get index stats
            try:
                index_connection = pc.Index(index.name)
                stats = index_connection.describe_index_stats()
                print(f"   Total Vectors: {stats.total_vector_count}")
                if stats.namespaces:
                    print(f"   Namespaces: {list(stats.namespaces.keys())}")
                print("-" * 40)
            except Exception as e:
                print(f"   Error getting stats: {str(e)}")
                print("-" * 40)
    
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {str(e)}")
        
        # Check if it's an authentication issue
        if "unauthorized" in str(e).lower() or "invalid" in str(e).lower():
            print("\n💡 Possible issues:")
            print("   - Check if your PINECONE_API_KEY is correct")
            print("   - Make sure your Pinecone account is active")
            print("   - Verify the API key has proper permissions")

if __name__ == "__main__":
    list_pinecone_indexes()
