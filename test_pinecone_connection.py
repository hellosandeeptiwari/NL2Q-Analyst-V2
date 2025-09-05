#!/usr/bin/env python3
"""Test Pinecone connection directly"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_pinecone_connection():
    """Test direct Pinecone connection with detailed diagnostics"""
    
    print("🔍 Testing Pinecone connection...")
    print("📋 Connection parameters:")
    
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    print(f"  API Key: {'*' * (len(api_key) - 8) + api_key[-8:] if api_key else 'None'}")
    print(f"  Environment: {environment}")
    print(f"  Index Name: {index_name}")
    
    try:
        import pinecone
        from pinecone import Pinecone
        
        print("\n🔌 Attempting to connect...")
        pc = Pinecone(api_key=api_key)
        print("✅ Pinecone client initialized!")
        
        print("\n📊 Testing index access...")
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"✅ Index access successful!")
        print(f"  Total vectors: {stats.total_vector_count}")
        print(f"  Dimension: {stats.dimension}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Additional diagnostics
        if "Failed to resolve" in str(e):
            print("💡 DNS resolution error usually indicates:")
            print("   - Network connectivity issues")
            print("   - Firewall blocking DNS resolution")
            print("   - Corporate proxy interfering with connections")
        elif "api_key" in str(e).lower():
            print("💡 API key error usually indicates:")
            print("   - Incorrect or expired API key")
            print("   - API key format issues")
        
        return False

if __name__ == "__main__":
    test_pinecone_connection()
