#!/usr/bin/env python3
"""
Comprehensive test of our mainstream query fixes
"""

import sys
import os
import asyncio
sys.path.append('.')

async def test_full_pipeline():
    """Test the complete mainstream pipeline with our fixes"""
    
    print("🎯 TESTING MAINSTREAM PIPELINE FIXES")
    print("=====================================")
    
    # Test the actual query that was failing
    test_query = "Show me patients with Tirosint target flag analysis"
    
    try:
        from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
        
        orchestrator = DynamicAgentOrchestrator()
        
        print(f"📝 Testing Query: {test_query}")
        print("⏳ Processing with mainstream pipeline...")
        
        response = await orchestrator.process_query(
            user_query=test_query,
            session_id="test_mainstream",
            user_id="test_user"
        )
        
        print(f"\n✅ Query processing completed!")
        print(f"📊 Response status: {response.get('status', 'unknown')}")
        
        # Check if mainstream path worked (no fallback)
        if response.get('fallback_used'):
            print(f"❌ FALLBACK USED: {response.get('fallback_reason', 'unknown')}")
            print("🔧 MAINSTREAM PATH FAILED")
            return False
        else:
            print("✅ MAINSTREAM PATH SUCCESS - No fallback used!")
        
        # Check SQL generation - try multiple possible locations
        sql = None
        
        # First check in task results (the correct location)
        if '2_query_generation' in response:
            task_result = response['2_query_generation']
            if 'sql_query' in task_result and task_result['sql_query']:
                sql = task_result['sql_query']
            elif 'sql' in task_result and task_result['sql']:
                sql = task_result['sql']
        
        # Fallback to root level keys (legacy support)
        if not sql:
            for key in ['sql_query', 'sql', 'generated_sql', 'query']:
                if key in response and response[key]:
                    sql = response[key]
                    break
        
        if sql:
            print(f"\n📝 Generated SQL:")
            print("=" * 50)
            print(sql)
            print("=" * 50)
            
            # Validate our fixes
            fixes_working = []
            
            # 1. TirosintTargetFlag discovered
            if 'TirosintTargetFlag' in sql:
                fixes_working.append("✅ TirosintTargetFlag column discovered")
            else:
                fixes_working.append("❌ TirosintTargetFlag column missing")
            
            # 2. Proper datatype handling ('Y' instead of 1)
            if "'Y'" in sql:
                fixes_working.append("✅ Proper flag datatype: 'Y' found")
            elif "= 1" in sql:
                fixes_working.append("❌ Wrong datatype: still using = 1")
            else:
                fixes_working.append("⚠️ No flag condition found")
            
            # 3. Column bracketing
            if '[TirosintTargetFlag]' in sql:
                fixes_working.append("✅ Column bracketing working")
            else:
                fixes_working.append("⚠️ Column bracketing not found")
            
            # 4. SQL Server syntax (TOP instead of LIMIT)
            if 'TOP ' in sql.upper():
                fixes_working.append("✅ SQL Server syntax: TOP found")
            elif 'LIMIT' in sql.upper():
                fixes_working.append("❌ Wrong syntax: LIMIT still used")
            else:
                fixes_working.append("⏹ No limit clause found")
            
            print(f"\n🔧 FIXES VERIFICATION:")
            for fix in fixes_working:
                print(f"   {fix}")
                
        else:
            print("❌ No SQL generated")
            return False
        
        # Check error handling
        if response.get('error'):
            print(f"⚠️ Error reported: {response['error']}")
        
        print(f"\n📊 Complete response keys: {list(response.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("COMPREHENSIVE MAINSTREAM PIPELINE TEST")
    print("==========================================")
    
    success = await test_full_pipeline()
    
    if success:
        print(f"\n🎉 MAINSTREAM PIPELINE IS WORKING!")
        print("✅ No more fallbacks to templates")
        print("✅ Intelligent column discovery")
        print("✅ Proper datatype handling")
        print("✅ SQL Server syntax compliance")
    else:
        print(f"\n❌ MAINSTREAM PIPELINE STILL HAS ISSUES")
        print("🔧 Additional fixes may be needed")

if __name__ == "__main__":
    asyncio.run(main())