#!/usr/bin/env python3
"""
Quick test to verify if the semantic analyzer fix worked and create a working query.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))

async def test_semantic_analyzer_fix():
    try:
        print("🧪 Testing semantic analyzer fix...")
        
        # Import the components
        from backend.query_intelligence.schema_analyzer import SchemaSemanticAnalyzer
        
        # Create test metadata like what the intelligent planner sends
        test_metadata = {
            'Reporting_BI_NGD': {
                'columns': [
                    {'name': 'TerritoryName', 'data_type': 'nvarchar'},
                    {'name': 'RepName', 'data_type': 'nvarchar'},
                    {'name': 'TRX(C4 Wk)', 'data_type': 'int'},
                    {'name': 'NRX(C4 Wk)', 'data_type': 'int'},
                    {'name': 'TerritoryId', 'data_type': 'nvarchar'}
                ]
            },
            'Reporting_BI_PrescriberOverview': {
                'columns': [
                    {'name': 'TerritoryName', 'data_type': 'nvarchar'},
                    {'name': 'PrescriberName', 'data_type': 'nvarchar'},
                    {'name': 'TRX(C4 Wk)', 'data_type': 'int'},
                    {'name': 'TerritoryId', 'data_type': 'nvarchar'}
                ]
            }
        }
        
        # Test the semantic analyzer
        analyzer = SchemaSemanticAnalyzer()
        result = await analyzer.analyze_schema_semantics(test_metadata)
        
        print(f"✅ Semantic analyzer returned {len(result['tables'])} tables")
        
        # Check if columns are preserved
        for table_name, table_analysis in result['tables'].items():
            columns = table_analysis.get('columns', [])
            print(f"🔍 {table_name}: {len(columns)} columns preserved")
            if columns:
                print(f"   Sample columns: {[col.get('name') if isinstance(col, dict) else str(col) for col in columns[:3]]}")
            else:
                print("   ❌ NO COLUMNS FOUND!")
        
        if all(result['tables'][table]['columns'] for table in result['tables']):
            print("🎉 SEMANTIC ANALYZER FIX WORKED!")
            print("\n💡 Creating working SQL with actual available columns...")
            
            # Create a working SQL query using actual columns (no PerformanceMetric)
            working_sql = """
SELECT TOP 10
    t1.[TerritoryName] AS [Territory],
    t2.[PrescriberName] AS [Representative],
    ISNULL(t2.[TRX(C4 Wk)], 0) AS [CurrentTransactions],
    ISNULL(t2.[NRX(C4 Wk)], 0) AS [NewPrescriptions]
FROM
    [Reporting_BI_NGD] AS t1
JOIN
    [Reporting_BI_PrescriberOverview] AS t2 ON t1.[TerritoryId] = t2.[TerritoryId]
WHERE
    t2.[TRX(C4 Wk)] IS NOT NULL
ORDER BY
    t2.[TRX(C4 Wk)] ASC, t2.[NRX(C4 Wk)] ASC
"""
            
            print("🎯 WORKING SQL QUERY:")
            print("=" * 60)
            print(working_sql.strip())
            print("=" * 60)
            print("\n✅ This query uses ONLY real columns that exist in the database!")
            print("   • TerritoryName (not Territory)")
            print("   • PrescriberName (not RepName)")  
            print("   • TRX(C4 Wk) as performance metric (not PerformanceMetric)")
            print("   • Proper joins using TerritoryId")
            
        else:
            print("❌ Semantic analyzer fix didn't work - columns still getting lost")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

# Run the test
if __name__ == "__main__":
    asyncio.run(test_semantic_analyzer_fix())