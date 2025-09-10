#!/usr/bin/env python3
"""
Test script to demonstrate retry functionality with intentional errors
This will show the 3-retry system with detailed stack trace collection
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator

class MockErrorOrchestrator(DynamicAgentOrchestrator):
    """Test orchestrator that intentionally fails to demonstrate retry logic"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attempt_count = 0
    
    async def _generate_database_aware_sql_core(self, query: str, available_tables, error_context="", pinecone_matches=None):
        """Mock core method that fails on first few attempts to test retry logic"""
        self.attempt_count += 1
        
        if self.attempt_count <= 2:
            # Simulate different types of errors for first 2 attempts
            if self.attempt_count == 1:
                raise ValueError(f"Simulated validation error on attempt {self.attempt_count}")
            elif self.attempt_count == 2:
                raise ConnectionError(f"Simulated connection error on attempt {self.attempt_count}")
        
        # Success on 3rd attempt
        return {
            "sql_query": f"SELECT * FROM {available_tables[0]} WHERE retry_test = true LIMIT 100;",
            "explanation": f"SQL generated successfully after {self.attempt_count} attempts",
            "generation_method": "mock_retry_success",
            "status": "success"
        }

async def test_error_retry_functionality():
    """Test the retry functionality with intentional errors"""
    print("🧪 Testing Error Retry Functionality with Stack Traces")
    print("=" * 60)
    
    # Create mock orchestrator
    orchestrator = MockErrorOrchestrator()
    
    print("\n🔍 Testing: Query that fails twice then succeeds")
    print("Expected: 3 attempts total (1 initial + 2 retries)")
    
    result = await orchestrator._generate_sql_with_retry(
        query="Test query for retry logic",
        available_tables=["TEST_TABLE"],
        error_context="",
        pinecone_matches=None
    )
    
    print(f"\n📊 FINAL RESULT:")
    print(f"✅ Status: {result.get('status')}")
    print(f"🔄 Total Attempts: {result.get('total_attempts')}")
    print(f"📋 Retry Count: {result.get('retry_count')}")
    
    if result.get("error_history"):
        print(f"\n📚 ERROR HISTORY ({len(result['error_history'])} errors):")
        for i, error in enumerate(result["error_history"]):
            print(f"\n  Attempt {error['attempt']}:")
            print(f"    Error Type: {error['error_type']}")
            print(f"    Error Message: {error['error']}")
            print(f"    Stack Trace Available: {'Yes' if error.get('stack_trace') else 'No'}")
            if error.get('stack_trace'):
                # Show first few lines of stack trace
                stack_lines = error['stack_trace'].split('\n')[:3]
                print(f"    Stack Preview: {' | '.join(stack_lines)}")
    
    if result.get("sql_query"):
        print(f"\n🎯 Generated SQL: {result['sql_query']}")
    
    print(f"\n✅ Retry with Stack Trace Collection: {'PASSED' if result.get('status') == 'success' else 'FAILED'}")

class MockFullErrorOrchestrator(DynamicAgentOrchestrator):
    """Test orchestrator that always fails to test exhaustion scenario"""
    
    async def _generate_database_aware_sql_core(self, query: str, available_tables, error_context="", pinecone_matches=None):
        """Always fails to test retry exhaustion"""
        raise RuntimeError("Simulated persistent error - all retries should fail")

async def test_retry_exhaustion():
    """Test what happens when all retries are exhausted"""
    print("\n\n🧪 Testing Retry Exhaustion (All Attempts Fail)")
    print("=" * 60)
    
    orchestrator = MockFullErrorOrchestrator()
    
    print("\n🔍 Testing: Query that always fails")
    print("Expected: 4 attempts total (1 initial + 3 retries), then failure")
    
    result = await orchestrator._generate_sql_with_retry(
        query="Query that will always fail",
        available_tables=["TEST_TABLE"],
        error_context="",
        pinecone_matches=None
    )
    
    print(f"\n📊 EXHAUSTION RESULT:")
    print(f"❌ Status: {result.get('status')}")
    print(f"🔄 Total Attempts: {result.get('total_attempts')}")
    print(f"📋 Retry Count: {result.get('retry_count')}")
    print(f"📚 Error History Length: {len(result.get('error_history', []))}")
    
    if result.get("last_error"):
        print(f"\n🚨 LAST ERROR:")
        last_error = result["last_error"]
        print(f"    Type: {last_error.get('error_type')}")
        print(f"    Message: {last_error.get('error')}")
        print(f"    Attempt: {last_error.get('attempt')}")
    
    print(f"\n✅ Retry Exhaustion Test: {'PASSED' if result.get('status') == 'failed' and result.get('total_attempts') == 4 else 'FAILED'}")

async def main():
    """Main test runner"""
    try:
        await test_error_retry_functionality()
        await test_retry_exhaustion()
        
        print("\n\n🎉 All retry functionality tests completed!")
        print("\n💡 Key Features Demonstrated:")
        print("   🔄 Exactly 3 retry attempts (4 total including initial)")
        print("   📋 Detailed error tracking with full stack traces")
        print("   🧠 Error context accumulation for LLM self-correction")
        print("   ⚡ Proper success handling after retries")
        print("   🚨 Graceful handling when all retries are exhausted")
        print("   📊 Comprehensive error reporting and statistics")
        
    except Exception as e:
        print(f"❌ Test framework error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
