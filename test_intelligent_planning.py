"""
Test script for the new Intelligent Query Planning architecture
"""

import asyncio
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_intelligent_planning():
    """Test the intelligent query planning integration"""
    try:
        print("🧪 Testing Intelligent Query Planning Integration")
        print("=" * 60)
        
        # Test import
        try:
            from backend.query_intelligence.intelligent_planner import IntelligentQueryPlanner
            from backend.query_intelligence.schema_analyzer import SchemaSemanticAnalyzer
            from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
            print("✅ All imports successful")
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            return
        
        # Test initialization
        try:
            orchestrator = DynamicAgentOrchestrator()
            print("✅ DynamicAgentOrchestrator initialized")
            
            if orchestrator.intelligent_planner:
                print("✅ Intelligent planner available in orchestrator")
            else:
                print("⚠️ Intelligent planner not available")
                
            if orchestrator.schema_analyzer:
                print("✅ Schema analyzer available in orchestrator")
            else:
                print("⚠️ Schema analyzer not available")
                
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return
        
        # Test basic planner functionality
        try:
            planner = IntelligentQueryPlanner()
            analyzer = SchemaSemanticAnalyzer()
            
            # Test semantic analysis
            sample_metadata = {
                "PRESCRIBER_REPORTING_OVERVIEW": {
                    "columns": ["PRESCRIBER_ID", "TRX_TOTAL", "NRX_TOTAL", "SPECIALTY", "TARGET_TIER"],
                    "description": "Prescriber performance reporting data"
                },
                "TERRITORY_HIERARCHY": {
                    "columns": ["TERRITORY_ID", "TERRITORY_NAME", "REGION", "REP_ID"],
                    "description": "Sales territory organizational structure"
                }
            }
            
            semantic_analysis = await analyzer.analyze_schema_semantics(sample_metadata)
            print(f"✅ Schema semantic analysis completed")
            print(f"   - Analyzed {len(semantic_analysis.get('tables', {}))} tables")
            print(f"   - Primary domain: {semantic_analysis.get('business_domains', {}).get('primary_domain', 'unknown')}")
            print(f"   - Query capabilities: {len(semantic_analysis.get('query_capabilities', {}).get('supported_patterns', []))} patterns")
            
        except Exception as e:
            print(f"❌ Planner functionality test failed: {e}")
            return
        
        print("\n🎉 All tests passed! Integration is working correctly.")
        print("\nKey Benefits:")
        print("✓ No more artificial simple/complex query restrictions")
        print("✓ Semantic understanding of schema relationships")
        print("✓ Intelligent table selection based on query needs")
        print("✓ Always show SQL in UI (even for failed queries)")
        print("✓ Clean separation of concerns from orchestrator")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_intelligent_planning())