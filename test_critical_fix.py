#!/usr/bin/env python3
"""
Simple test to verify the critical string/dict error is completely fixed
"""

import asyncio

async def test_critical_fix():
    print("Testing critical string/dict error fix...")
    
    try:
        from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
        
        # Initialize orchestrator
        orchestrator = DynamicAgentOrchestrator()
        
        # Test with the query that was causing the error
        result = await orchestrator.process_query(
            "Show me 5 prescribers",
            user_id="critical_test_user",
            session_id="critical_test_session",
            use_deterministic=False
        )
        
        # Check results
        status = result.get('status', 'unknown')
        print(f"Final Status: {status}")
        
        if status == 'completed':
            print("SUCCESS: System completed without critical string/dict errors")
            data_count = len(result.get('results', {}).get('data', []))
            print(f"Data Retrieved: {data_count} rows")
            return True
        else:
            print("FAILURE: System did not complete successfully")
            return False
            
    except Exception as e:
        if "'str' object has no attribute 'get'" in str(e):
            print(f"FAILURE: Critical string/dict error still exists: {e}")
            return False
        else:
            print(f"OTHER ERROR: {e}")
            return False

if __name__ == "__main__":
    success = asyncio.run(test_critical_fix())
    if success:
        print("\nCRITICAL FIX VERIFICATION: PASSED")
    else:
        print("\nCRITICAL FIX VERIFICATION: FAILED")