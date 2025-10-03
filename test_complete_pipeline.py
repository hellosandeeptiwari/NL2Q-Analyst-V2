#!/usr/bin/env python3
"""
Test the complete query pipeline with our fixes
"""

import sys
import os
import asyncio
sys.path.append('.')

from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
from backend.query_intelligence.intelligent_query_planner import IntelligentQueryPlanner

async def test_complete_pipeline():
    """Test the complete query pipeline to verify our fixes"""
    
    print("🧪 Testing complete NL2Q pipeline with our fixes...")
    
    # Test query that should use TirosintTargetFlag
    test_query = "Show me patients with Tirosint target flag analysis"
    
    print(f"📝 Query: {test_query}")
    
    # Initialize orchestrator which includes the query planner
    orchestrator = DynamicAgentOrchestrator()
    
    try:
        print("\n🔍 Testing SQL generation with our fixes...")
        
        response = await orchestrator.process_query(
            user_query=test_query,
            session_id="test_session",
            user_id="test_user"
        )
        
        print(f"✅ Query processing completed")
        print(f"   Status: {response.get('status', 'unknown')}")
        
        if 'sql_query' in response:
            sql = response['sql_query']
            print(f"   Generated SQL: {sql[:200]}...")
            
            # Check for our fixes
            if 'TOP ' in sql.upper():
                print("   ✅ LIMIT to TOP conversion working")
            else:
                print("   ⚠️ LIMIT to TOP conversion may not have been needed")
                
            if '[TirosintTargetFlag]' in sql:
                print("   ✅ Column bracketing working")
            elif 'TirosintTargetFlag' in sql:
                print("   ⚠️ TirosintTargetFlag found but not bracketed")
            else:
                print("   ❌ TirosintTargetFlag column not found in SQL")
                
        else:
            print("   ❌ No SQL generated")
            
        if 'fallback_used' in response:
            if response['fallback_used']:
                print("   ⚠️ Fallback was used - this indicates an issue")
                print(f"   Fallback reason: {response.get('fallback_reason', 'unknown')}")
            else:
                print("   ✅ No fallback used - intelligent path worked")
        
        # Check if our hardcoded schema fix is working
        if 'schema_info' in response or 'columns_used' in response:
            print("   ✅ Schema information included in response")
        
        # Print more details for debugging
        if response.get('error'):
            print(f"   ❌ Error: {response['error']}")
            
        print(f"\n📊 Full response keys: {list(response.keys())}")
        
        # Show full SQL if it contains TirosintTargetFlag
        if 'sql_query' in response and 'TirosintTargetFlag' in response['sql_query']:
            print(f"\n📝 Full SQL with TirosintTargetFlag:")
            print(response['sql_query'])
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_complete_pipeline())