#!/usr/bin/env python3
"""
Setup Azure AI Search for Schema Vector Storage
Run this after adding Azure AI Search credentials to .env file
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

async def setup_azure_search_integration():
    """Setup Azure AI Search for schema vector storage"""
    
    print("🚀 Setting up Azure AI Search Integration for Schema Vector Storage")
    print("=" * 80)
    
    # Check required environment variables
    required_vars = {
        "AZURE_SEARCH_ENDPOINT": "Your Azure AI Search service endpoint (e.g., https://yourservice.search.windows.net)",
        "AZURE_SEARCH_KEY": "Your Azure AI Search admin key",
        "AZURE_SEARCH_INDEX_NAME": "Name for your search index (e.g., 'nl2q-schema-index')",
        "OPENAI_API_KEY": "Your OpenAI API key for embeddings",
        "OPENAI_EMBEDDING_MODEL": "OpenAI embedding model (default: text-embedding-3-large)"
    }
    
    print("🔍 Checking environment variables...")
    missing_vars = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if var == "OPENAI_EMBEDDING_MODEL" and not value:
            value = "text-embedding-3-large"  # Default
        
        if not value:
            missing_vars.append((var, description))
            print(f"   ❌ {var}: Missing")
        else:
            if "KEY" in var or "PASSWORD" in var:
                print(f"   ✅ {var}: ***hidden***")
            else:
                print(f"   ✅ {var}: {value}")
    
    if missing_vars:
        print(f"\n⚠️  Missing required environment variables:")
        print(f"\nPlease add the following to your .env file:")
        print("-" * 50)
        for var, description in missing_vars:
            print(f"{var}=your_value_here  # {description}")
        print("-" * 50)
        print(f"\nAzure AI Search Setup Instructions:")
        print(f"1. Go to Azure Portal (portal.azure.com)")
        print(f"2. Create an Azure AI Search service")
        print(f"3. Get the service endpoint URL and admin key")
        print(f"4. Choose a descriptive index name (e.g., 'nl2q-schema-index')")
        print(f"5. Add the credentials to your .env file")
        print(f"6. Run this script again")
        return False
    
    print(f"\n✅ All environment variables configured!")
    
    try:
        # Initialize Azure Search integration
        from backend.azure_schema_vector_store import AzureSchemaVectorStore
        
        print(f"\n🔧 Initializing Azure AI Search...")
        vector_store = AzureSchemaVectorStore()
        
        # Create search index
        print(f"📝 Creating search index...")
        await vector_store.create_search_index()
        
        # Test with database connection
        print(f"\n📊 Testing database connection...")
        from backend.db.engine import get_adapter
        db_adapter = get_adapter("snowflake")
        
        # Index database schema
        print(f"🚀 Starting schema indexing process...")
        chunk_count = await vector_store.index_database_schema(db_adapter)
        
        print(f"\n✅ Schema indexing complete!")
        print(f"   Total chunks indexed: {chunk_count}")
        
        # Test search functionality
        print(f"\n🔍 Testing search functionality...")
        test_queries = [
            "read table final nba output python",
            "nba data analysis",
            "provider input data",
            "similarity output results"
        ]
        
        for query in test_queries:
            print(f"\n   Testing: '{query}'")
            results = await vector_store.search_relevant_tables(query, top_k=4)
            
            print(f"   Found {len(results)} relevant tables:")
            for i, table in enumerate(results):
                print(f"      {i+1}. {table['table_name']} (score: {table['best_chunk_score']:.3f})")
        
        print(f"\n🎉 Azure AI Search setup complete!")
        print(f"✅ Schema vector storage is ready for use")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_search_with_user_query():
    """Test search with the user's specific query"""
    
    print(f"\n🧪 Testing with user's NBA query...")
    
    try:
        from backend.azure_schema_vector_store import AzureSchemaVectorStore
        
        vector_store = AzureSchemaVectorStore()
        
        user_query = "read table final nba output python and fetch top 5 rows and create a visualization with frequency of recommended message and provider input"
        
        print(f"🔍 User Query: {user_query}")
        print(f"🔍 Searching for top 4 relevant tables...")
        
        results = await vector_store.search_relevant_tables(user_query, top_k=4)
        
        print(f"\n📊 TOP 4 TABLE SUGGESTIONS:")
        print("=" * 60)
        
        for i, table in enumerate(results):
            print(f"\n{i+1}. TABLE: {table['table_name']}")
            print(f"   Relevance Score: {table['best_chunk_score']:.3f}")
            print(f"   Chunk Types: {', '.join(table['chunk_types'])}")
            print(f"   Sample Content: {table['sample_content']}")
        
        if len(results) >= 4:
            print(f"\n✅ Successfully found 4 table suggestions!")
            print(f"🎯 User can now select from these options")
        else:
            print(f"\n⚠️  Only found {len(results)} tables - may need more indexing")
        
        return results
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return []

if __name__ == "__main__":
    print("Azure AI Search Setup for NL2Q Schema Vector Storage")
    print("=" * 60)
    
    # Run setup
    setup_success = asyncio.run(setup_azure_search_integration())
    
    if setup_success:
        # Test with user query
        asyncio.run(test_search_with_user_query())
        
        print(f"\n🎉 SETUP COMPLETE!")
        print(f"Your NL2Q system now has intelligent schema discovery with:")
        print(f"✅ Azure AI Search vector storage")
        print(f"✅ OpenAI large embeddings") 
        print(f"✅ Top 4 table suggestions for user selection")
        print(f"✅ Improved similarity matching")
    else:
        print(f"\n❌ Setup incomplete - please configure environment variables")
