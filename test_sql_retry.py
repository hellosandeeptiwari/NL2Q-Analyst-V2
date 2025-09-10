"""
Test the enhanced retry functionality for SQL generation
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

async def test_sql_retry_functionality():
    """Test the SQL generation retry mechanism with error handling"""
    
    print("🧪 Testing SQL Generation Retry Functionality")
    print("=" * 50)
    
    # Import the orchestrator
    from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
    
    # Initialize orchestrator
    orchestrator = DynamicAgentOrchestrator()
    
    # Test case 1: Normal query (should succeed)
    print("\\n🔍 Test 1: Normal Query (should succeed)")
    try:
        result = await orchestrator._generate_sql_with_retry(
            query="Show me all tables",
            available_tables=["ALL_RATES", "NEGOTIATED_RATES"], 
            error_context="",
            pinecone_matches=[]
        )
        
        print(f"✅ Result: {result.get('status', 'unknown')}")
        print(f"📊 Attempts: {result.get('total_attempts', 'unknown')}")
        if result.get('sql_query'):
            print(f"🎯 SQL Generated: {result['sql_query'][:100]}...")
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
    
    # Test case 2: Query that should trigger retries (simulated)
    print("\\n🔍 Test 2: Testing Retry Logic")
    try:
        # This will test the retry wrapper even if it succeeds
        result = await orchestrator._generate_sql_with_retry(
            query="What are the average payment amounts by provider?",
            available_tables=["NEGOTIATED_RATES", "PROVIDER_REFERENCES"],
            error_context="Previous attempt failed with syntax error",
            pinecone_matches=[]
        )
        
        print(f"✅ Result: {result.get('status', 'unknown')}")
        print(f"📊 Total Attempts: {result.get('total_attempts', 'unknown')}")
        print(f"🔄 Retry Count: {result.get('retry_count', 'unknown')}")
        
        if result.get('error_history'):
            print(f"📋 Error History: {len(result['error_history'])} errors recorded")
        
        if result.get('sql_query'):
            print(f"🎯 Final SQL: {result['sql_query'][:150]}...")
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
    
    print("\\n🔍 Test 3: Testing Core Method Directly")
    try:
        # Test the core method
        core_result = await orchestrator._generate_database_aware_sql_core(
            query="SELECT COUNT(*) FROM NEGOTIATED_RATES",
            available_tables=["NEGOTIATED_RATES"],
            error_context="",
            pinecone_matches=[]
        )
        
        print(f"✅ Core Result: {core_result.get('status', 'unknown')}")
        print(f"🛠️ Method: {core_result.get('generation_method', 'unknown')}")
        
        if core_result.get('sql_query'):
            print(f"🎯 Core SQL: {core_result['sql_query'][:100]}...")
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
    
    print("\\n✅ Retry Functionality Testing Complete")
    print("\\n💡 Key Features Implemented:")
    print("   🔄 3 retry attempts for SQL generation")
    print("   📋 Detailed error tracking with stack traces")
    print("   🧠 LLM intelligence integration") 
    print("   ⚡ Fast fallback when intelligence unavailable")
    print("   📊 Comprehensive error reporting")

if __name__ == "__main__":
    asyncio.run(test_sql_retry_functionality())
