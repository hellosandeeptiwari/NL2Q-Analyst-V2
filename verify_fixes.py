"""
Verification script for database connection fixes
Run this to verify all changes are working correctly
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("🔍 VERIFICATION SCRIPT FOR DATABASE CONNECTION FIXES")
print("=" * 80)

def check_environment_variables():
    """Check if required environment variables are set"""
    print("\n📋 Step 1: Checking Environment Variables...")
    
    required_vars = {
        "DB_ENGINE": "Database engine type",
        "OPENAI_API_KEY": "OpenAI API key for LLM features",
        "PINECONE_API_KEY": "Pinecone API key for vector search"
    }
    
    optional_vars = {
        "AZURE_HOST": "Azure SQL Server host",
        "AZURE_DATABASE": "Azure SQL database name",
        "SNOWFLAKE_ACCOUNT": "Snowflake account",
        "POSTGRES_HOST": "PostgreSQL host"
    }
    
    missing_required = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if "KEY" in var or "PASSWORD" in var:
                display_value = "***" + value[-4:] if len(value) > 4 else "***"
            else:
                display_value = value
            print(f"  ✅ {var} = {display_value}")
        else:
            print(f"  ❌ {var} is missing ({description})")
            missing_required.append(var)
    
    print("\n  Optional variables:")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            if "KEY" in var or "PASSWORD" in var:
                display_value = "***" + value[-4:] if len(value) > 4 else "***"
            else:
                display_value = value
            print(f"  ℹ️  {var} = {display_value}")
    
    if missing_required:
        print(f"\n  ⚠️  Missing required variables: {', '.join(missing_required)}")
        return False
    
    print("\n  ✅ All required environment variables are set")
    return True

async def test_database_adapter():
    """Test database adapter initialization"""
    print("\n📋 Step 2: Testing Database Adapter...")
    
    try:
        from backend.db.engine import get_adapter
        
        db_engine = os.getenv("DB_ENGINE", "azure")
        print(f"  🔍 Using DB_ENGINE: {db_engine}")
        
        adapter = get_adapter(db_engine)
        
        if adapter is None:
            print(f"  ❌ Database adapter returned None for engine '{db_engine}'")
            print(f"  💡 Make sure {db_engine.upper()} connection variables are set")
            return False
        
        print(f"  ✅ Database adapter initialized: {type(adapter).__name__}")
        
        # Test connection
        print(f"  🔍 Testing database connection...")
        result = adapter.run("SELECT 1 as test", dry_run=False)
        
        if result and not result.error:
            print(f"  ✅ Database connection test successful")
            return True
        else:
            error_msg = result.error if result else "No result returned"
            print(f"  ❌ Database connection test failed: {error_msg}")
            return False
            
    except ImportError as e:
        print(f"  ❌ Failed to import database adapter: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Database adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_pinecone_connection():
    """Test Pinecone vector store connection"""
    print("\n📋 Step 3: Testing Pinecone Connection...")
    
    try:
        from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
        
        store = PineconeSchemaVectorStore()
        
        if store is None:
            print(f"  ❌ Pinecone store returned None")
            return False
        
        print(f"  ✅ Pinecone store initialized")
        
        # Test connection
        print(f"  🔍 Testing Pinecone index...")
        stats = store.index.describe_index_stats()
        vector_count = stats.total_vector_count
        
        print(f"  ✅ Pinecone connection successful")
        print(f"  📊 Current vectors in index: {vector_count}")
        
        if vector_count == 0:
            print(f"  ⚠️  Index is empty - auto-indexing should run on startup")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Failed to import Pinecone store: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Pinecone connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_orchestrator_initialization():
    """Test orchestrator initialization"""
    print("\n📋 Step 4: Testing Orchestrator Initialization...")
    
    try:
        from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
        
        orchestrator = DynamicAgentOrchestrator()
        print(f"  ✅ Orchestrator instantiated")
        
        print(f"  🔍 Running initialization...")
        await orchestrator.initialize_on_startup()
        
        # Check if database connector was initialized
        if orchestrator.db_connector is None:
            print(f"  ❌ Database connector is None after initialization")
            return False
        else:
            print(f"  ✅ Database connector initialized")
        
        # Check if Pinecone store was initialized
        if orchestrator.pinecone_store is None:
            print(f"  ⚠️  Pinecone store is None (may be expected if connection failed)")
        else:
            print(f"  ✅ Pinecone store initialized")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Failed to import orchestrator: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Orchestrator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_all_tests():
    """Run all verification tests"""
    print("\n🚀 Starting verification tests...\n")
    
    results = {
        "environment": False,
        "database": False,
        "pinecone": False,
        "orchestrator": False
    }
    
    # Step 1: Environment variables
    results["environment"] = check_environment_variables()
    
    if not results["environment"]:
        print("\n⚠️  Cannot proceed without required environment variables")
        return results
    
    # Step 2: Database adapter
    results["database"] = await test_database_adapter()
    
    # Step 3: Pinecone connection
    results["pinecone"] = await test_pinecone_connection()
    
    # Step 4: Orchestrator initialization
    results["orchestrator"] = await test_orchestrator_initialization()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 80)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test.replace('_', ' ').title()}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - System is ready!")
        print("=" * 80)
        print("\n💡 Next steps:")
        print("  1. Start the application")
        print("  2. Try a test query: 'Show me top 10 prescribers'")
        print("  3. Monitor logs for any issues")
    else:
        print("⚠️  SOME TESTS FAILED - Please review errors above")
        print("=" * 80)
        print("\n💡 Common fixes:")
        print("  - Check your .env file has all required variables")
        print("  - Verify database credentials are correct")
        print("  - Ensure Pinecone API key is valid")
        print("  - Make sure DB_ENGINE matches your setup (azure/snowflake/postgres)")
    
    print()
    
    return results

if __name__ == "__main__":
    # Load environment variables from .env file if it exists
    try:
        from dotenv import load_dotenv
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✅ Loaded environment variables from {env_path}")
        else:
            print(f"⚠️  No .env file found at {env_path}")
    except ImportError:
        print("⚠️  python-dotenv not installed, using system environment variables only")
    
    # Run tests
    asyncio.run(run_all_tests())
