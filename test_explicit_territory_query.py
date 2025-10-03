#!/usr/bin/env python3
"""
Test the specific territory query that should require JOINs
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from backend.query_intelligence.intelligent_query_planner import IntelligentQueryPlanner

async def test_territory_query():
    """Test the specific territory underperformance query"""
    
    print("🧪 Testing territory underperformance query for JOIN detection...")
    
    planner = IntelligentQueryPlanner()
    
    # More explicit query that clearly requires rep names from different table
    query = "Show me territories with high rep activity but low prescriptions. I need the rep names and territory names and prescription counts and call counts."
    
    # Available tables with actual column structure
    available_tables = [
        {
            'table_name': 'Reporting_BI_PrescriberOverview',
            'columns': [
                'TerritoryName', 'TerritoryId', 'PrescriberName', 'PrescriberId',
                'TRX(C4 Wk)', 'Calls4', 'TirosintTargetFlag', 'Calls13', 'CallsQTD'
            ]
        },
        {
            'table_name': 'Reporting_BI_NGD', 
            'columns': [
                'RepName', 'TerritoryName', 'TerritoryId', 'PrescriberId',
                'TotalCalls', 'RegionName'
            ]
        }
    ]
    
    print(f"🔍 Query: {query}")
    print(f"📊 Available tables: {len(available_tables)}")
    
    try:
        # Test semantic analysis
        semantics = planner._extract_query_semantics(query, available_tables)
        
        print(f"\n🎯 Results:")
        print(f"   Requires JOIN: {semantics.get('requires_join', False)}")
        print(f"   Single table sufficient: {semantics.get('single_table_sufficient', False)}")
        
        join_reasons = semantics.get('join_reasons', [])
        if join_reasons:
            print(f"   JOIN reasons:")
            for reason in join_reasons:
                print(f"     - {reason}")
        
        # Test data requirement extraction
        query_lower = query.lower()
        cross_table_analysis = planner._analyze_cross_table_requirements(query_lower, available_tables)
        
        print(f"\n📋 Cross-table analysis:")
        print(f"   Requires JOIN: {cross_table_analysis.get('requires_join', False)}")
        print(f"   Single table sufficient: {cross_table_analysis.get('single_table_sufficient', False)}")
        
        table_coverage = cross_table_analysis.get('table_coverage', {})
        if table_coverage:
            print(f"   Table coverage analysis:")
            for table, coverage in table_coverage.items():
                percentage = coverage.get('coverage_percentage', 0) * 100
                satisfied = coverage.get('satisfied_requirements', [])
                print(f"     {table}: {percentage:.1f}% coverage")
                if satisfied:
                    print(f"       Satisfied: {', '.join(satisfied[:5])}")  # Show first 5
        
        # Check if this would trigger a JOIN
        if semantics.get('requires_join') or cross_table_analysis.get('requires_join'):
            print(f"\n✅ SUCCESS: Query correctly detected as requiring JOINs!")
        else:
            print(f"\n⚠️  WARNING: Query not detected as requiring JOINs")
            
    except Exception as e:
        print(f"💥 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_territory_query())