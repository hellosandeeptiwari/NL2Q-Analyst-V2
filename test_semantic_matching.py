#!/usr/bin/env python3
"""
Test semantic matching for JOIN detection
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from backend.query_intelligence.intelligent_query_planner import IntelligentQueryPlanner

async def test_semantic_matching():
    """Test semantic matching for JOIN detection"""
    
    print("🧪 Testing semantic matching for JOIN detection...")
    
    # Initialize the planner
    planner = IntelligentQueryPlanner()
    
    # Test queries
    test_cases = [
        {
            "query": "Show me territories where we have good rep coverage but low prescriptions. Include rep names.",
            "expected_join": True,
            "reason": "Rep names only in NGD table, other data in PrescriberOverview"
        },
        {
            "query": "List some prescribers with Tirosint target flag",
            "expected_join": False,
            "reason": "Simple single-table query pattern detected"
        },
        {
            "query": "Compare prescriber performance across territories with representative information",
            "expected_join": True,
            "reason": "Cross-table data requirements (prescriber + territory + rep data)"
        }
    ]
    
    # Mock available tables (simplified for testing)
    available_tables = [
        {
            'table_name': 'Reporting_BI_PrescriberOverview',
            'columns': [
                'TerritoryName', 'PrescriberName', 'TRX(C4 Wk)', 'Calls4', 
                'TirosintTargetFlag', 'PrescriberId', 'TerritoryId'
            ]
        },
        {
            'table_name': 'Reporting_BI_NGD', 
            'columns': [
                'RepName', 'TerritoryName', 'TerritoryId', 'PrescriberId'
            ]
        }
    ]
    
    print(f"🔍 Testing with {len(test_cases)} test cases...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected_join = test_case["expected_join"]
        reason = test_case["reason"]
        
        print(f"📋 Test Case {i}: {query[:60]}...")
        print(f"   Expected JOIN: {expected_join}")
        print(f"   Reason: {reason}")
        
        try:
            # Extract query semantics (this calls our semantic matching logic)
            semantics = planner._extract_query_semantics(query, available_tables)
            
            # Check the result
            actual_join = semantics.get('requires_join', False)
            join_reasons = semantics.get('join_reasons', [])
            
            print(f"   Detected JOIN: {actual_join}")
            if join_reasons:
                print(f"   JOIN Reasons: {', '.join(join_reasons)}")
            
            # Verify result
            if actual_join == expected_join:
                print(f"   ✅ PASSED - Correctly detected JOIN requirement")
            else:
                print(f"   ❌ FAILED - Expected {expected_join}, got {actual_join}")
                
        except Exception as e:
            print(f"   💥 ERROR: {str(e)}")
            
        print()
    
    print("🎯 Semantic matching test completed!")

if __name__ == "__main__":
    asyncio.run(test_semantic_matching())