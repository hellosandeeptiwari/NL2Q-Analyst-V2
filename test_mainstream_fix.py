#!/usr/bin/env python3
"""
Test the mainstream flow fix to verify LLM receives real columns
"""

import asyncio
import sys
sys.path.append('.')

async def test_mainstream_fix():
    print("🎯 TESTING MAINSTREAM COLUMN FIX")
    print("=" * 50)
    
    # Test the full mainstream flow
    from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
    
    try:
        orchestrator = DynamicAgentOrchestrator()
        
        # Test query that previously failed with fake columns
        query = "Show me territories with the lowest performance, including rep names"
        print(f"📝 Query: {query}")
        
        # This should now use REAL columns from the database
        print("\n🔄 Running mainstream SQL generation...")
        print("   (Watch for debug output showing real columns)")
        
        result = await orchestrator.process_query(query)
        
        print(f"\n📊 Result status: {result.get('status')}")
        if result.get('sql'):
            print(f"🔍 Generated SQL:\n{result['sql']}")
            
            # Check if SQL uses real columns
            real_columns = ['TerritoryName', 'PrescriberName', 'TRX(C4 Wk)', 'NRX(C4 Wk)']
            fake_columns = ['Territory', 'RepName', 'PerformanceMetric', 'RepID']
            
            sql = result['sql']
            
            print(f"\n✅ REAL COLUMN USAGE:")
            for col in real_columns:
                if col in sql or col.replace('(', '\(').replace(')', '\)') in sql:
                    print(f"   ✅ {col} found in SQL")
                else:
                    print(f"   ⚠️  {col} not found in SQL")
                    
            print(f"\n❌ FAKE COLUMN CHECK:")
            found_fake = False
            for col in fake_columns:
                if col in sql and col not in ['TerritoryName', 'PrescriberName']:
                    print(f"   🚨 FAKE column {col} found in SQL!")
                    found_fake = True
                else:
                    print(f"   ✅ FAKE column {col} correctly absent")
                    
            if not found_fake:
                print(f"\n🎉 SUCCESS! Mainstream flow now uses only REAL columns!")
            else:
                print(f"\n❌ FAILURE! Mainstream flow still using fake columns!")
                
        else:
            print(f"⚠️ No SQL generated - check error: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Error testing mainstream fix: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mainstream_fix())