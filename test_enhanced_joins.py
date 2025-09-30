#!/usr/bin/env python3
"""
Test the enhanced join system with comprehensive few-shot examples
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.db.engine import get_adapter
from backend.query_intelligence.intelligent_query_planner import IntelligentQueryPlanner

async def test_enhanced_join_system():
    """Test the complete enhanced join system with realistic query"""
    
    print("🧪 Testing Enhanced Join System with Few-Shot Examples")
    print("=" * 60)
    
    # Connect to database
    os.environ['DB_ENGINE'] = 'azure_sql'  # Set the database engine
    db_adapter = get_adapter()
    print("✅ Connecting to Azure SQL Server...")
    print(f"✅ Connected successfully")
    
    # Initialize intelligent query planner
    planner = IntelligentQueryPlanner()
    
    # Test query that requires joins
    test_query = "summarize top 10 prescribers of tirosint sol by territory showing their prescription volume"
    
    # Mock context with ALL 3 confirmed tables (simulating the orchestrator)
    confirmed_tables = [
        'Reporting_BI_PrescriberProfile', 
        'Reporting_BI_PrescriberOverview',
        'Reporting_BI_NGD'  # Add the third table!
    ]
    
    # Create context similar to what orchestrator provides
    context = {
        'db_adapter': db_adapter,
        'matched_tables': [
            {'table_name': 'Reporting_BI_PrescriberProfile', 'columns': []},
            {'table_name': 'Reporting_BI_PrescriberOverview', 'columns': []},
            {'table_name': 'Reporting_BI_NGD', 'columns': []}  # Add the third table!
        ]
    }
    
    print(f"\n🎯 Test Query: {test_query}")
    print(f"📊 Tables: {confirmed_tables}")
    
    try:
        # Generate query with intelligent planning
        result = await planner.generate_query_with_plan(
            query=test_query,
            context=context,
            confirmed_tables=confirmed_tables
        )
        
        print(f"\n✅ RESULT SUMMARY:")
        print(f"   Confidence Score: {result.get('confidence_score', result.get('confidence', 'N/A'))}")
        print(f"   Tables Used: {result.get('tables_used', confirmed_tables)}")
        print(f"   Join Strategy: {len(result.get('join_strategy', []))} relationships discovered")
        print(f"   Business Rules: {len(result.get('business_logic_applied', []))} rules applied")
        
        if result.get('sql'):
            print(f"\n📝 GENERATED SQL:")
            print("=" * 50)
            print(result['sql'])
            print("=" * 50)
        
        if result.get('explanation'):
            print(f"\n💡 EXPLANATION:")
            print(result['explanation'])
        
        # Show confidence breakdown
        if result.get('query_plan_metadata', {}).get('confidence_breakdown'):
            confidence_breakdown = result['query_plan_metadata']['confidence_breakdown']
            print(f"\n📊 CONFIDENCE BREAKDOWN:")
            print(f"   Final Score: {confidence_breakdown['final_score']}")
            print(f"   Methodology: {confidence_breakdown['methodology']}")
            print(f"   Factors Considered: {confidence_breakdown['factors_considered']}")
        
        # Show join strategy details
        if result.get('join_strategy'):
            print(f"\n🔗 JOIN STRATEGY DETAILS:")
            for i, join in enumerate(result['join_strategy'][:3], 1):  # Show top 3
                print(f"   {i}. {join.get('table1', 'T1')} → {join.get('table2', 'T2')}")
                print(f"      Join: {join.get('join_columns', ['?', '?'])} (confidence: {join.get('confidence', 0.0):.2f})")
        
        # Show intelligent enhancements
        if result.get('intelligent_enhancements'):
            enhancements = result['intelligent_enhancements']
            print(f"\n🧠 INTELLIGENT ENHANCEMENTS:")
            print(f"   Multi-table Analysis: {enhancements.get('multi_table_analysis', False)}")
            print(f"   Semantic Join Discovery: {enhancements.get('semantic_join_discovery', False)}")
            print(f"   Business Context Applied: {enhancements.get('business_context_applied', False)}")
            print(f"   Schema Intelligence Used: {enhancements.get('schema_intelligence_used', False)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in enhanced join test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_enhanced_join_system())
    
    if result:
        print(f"\n🎉 Enhanced Join System Test COMPLETED")
        print(f"Final Confidence: {result.get('confidence_score', result.get('confidence', 'N/A'))}")
    else:
        print(f"\n❌ Enhanced Join System Test FAILED")