#!/usr/bin/env python3
"""
Test script for template SQL generation (unit tests)
Tests the intelligent template generation without needing the full server
"""

import sys
import os
sys.path.append(os.getcwd())

import asyncio
from backend.query_intelligence.intelligent_query_planner import IntelligentQueryPlanner
from backend.db.engine import get_adapter

async def test_template_generation():
    """Test the _generate_template_sql method directly"""
    print("🧪 Testing Template SQL Generation (Unit Tests)")
    print("=" * 80)
    
    try:
        # Initialize components
        planner = IntelligentQueryPlanner()
        planner.db_adapter = get_adapter()
        
        # Mock schema (based on actual Tirosint data structure)
        mock_schema = {
            'tables': {
                'Reporting_BI_PrescriberOverview': {
                    'columns': [
                        {'name': 'TerritoryName', 'data_type': 'varchar'},
                        {'name': 'RegionName', 'data_type': 'varchar'},
                        {'name': 'ProductGroupName', 'data_type': 'varchar'},
                        {'name': 'PrescriberName', 'data_type': 'varchar'},
                        {'name': 'TRX(C4 Wk)', 'data_type': 'int'},
                        {'name': 'TRX(C13 Wk)', 'data_type': 'int'},
                        {'name': 'NRX(C4 Wk)', 'data_type': 'int'},
                        {'name': 'TQTY(C4 Wk)', 'data_type': 'int'},
                        {'name': 'Specialty', 'data_type': 'varchar'},
                        {'name': 'Address', 'data_type': 'varchar'}
                    ]
                }
            }
        }
        
        # Test cases
        test_cases = [
            {
                'query': 'summarize top 10 sales of tirosint sol by territory',
                'expected_features': ['TOP 10', 'Tirosint', 'Territory', 'TRX', 'ORDER BY']
            },
            {
                'query': 'show me top 5 prescriptions of Levothyroxine by region',
                'expected_features': ['TOP 5', 'Levothyroxine', 'Region']
            },
            {
                'query': 'get top 15 records by sales volume', 
                'expected_features': ['TOP 15', 'ORDER BY', 'TRX']
            },
            {
                'query': 'display all prescriber data',
                'expected_features': ['TOP 100', 'Territory', 'Product']
            },
            {
                'query': 'find data for product "Synthroid" in territories',
                'expected_features': ['Synthroid', 'WHERE', 'Territory']
            },
            {
                'query': 'Get top records',  # Should NOT filter by "Get"
                'expected_features': ['TOP', 'no WHERE']
            }
        ]
        
        print("🎯 Running Template Generation Tests...\n")
        
        results = []
        for i, test_case in enumerate(test_cases, 1):
            query = test_case['query']
            expected = test_case['expected_features']
            
            print(f"📝 Test {i}: {query}")
            print("-" * 60)
            
            try:
                # Generate SQL using our improved template method
                sql = planner._generate_template_sql(
                    query, 
                    mock_schema, 
                    ['Reporting_BI_PrescriberOverview']
                )
                
                print(f"Generated SQL:")
                print(sql)
                print()
                
                # Feature analysis
                detected_features = []
                sql_upper = sql.upper()
                
                # Check TOP limits - order matters, check specific numbers first
                if 'TOP 100' in sql_upper:
                    detected_features.append('TOP 100')
                elif 'TOP 15' in sql_upper:
                    detected_features.append('TOP 15')
                elif 'TOP 10' in sql_upper:
                    detected_features.append('TOP 10')
                elif 'TOP 5' in sql_upper:
                    detected_features.append('TOP 5')
                elif 'TOP ' in sql_upper:
                    detected_features.append('Dynamic TOP')
                
                # Check filtering
                if 'WHERE' in sql_upper:
                    detected_features.append('WHERE clause')
                    if 'TIROSINT' in sql_upper:
                        detected_features.append('Tirosint filtering')
                    if 'LEVOTHYROXINE' in sql_upper:
                        detected_features.append('Levothyroxine filtering')
                    if 'SYNTHROID' in sql_upper:
                        detected_features.append('Synthroid filtering')
                else:
                    detected_features.append('no WHERE')
                
                # Check ordering
                if 'ORDER BY' in sql_upper:
                    detected_features.append('ORDER BY')
                    if 'TRX' in sql_upper:
                        detected_features.append('TRX ordering')
                
                # Check column selection
                if 'TERRITORYNAME' in sql_upper:
                    detected_features.append('Territory')
                if 'REGIONNAME' in sql_upper:
                    detected_features.append('Region')
                if 'TRX' in sql_upper:
                    detected_features.append('TRX')
                if 'PRODUCTGROUPNAME' in sql_upper:
                    detected_features.append('Product')
                
                print(f"✅ Detected Features: {', '.join(detected_features)}")
                
                # Check against expectations
                missing = []
                for exp in expected:
                    if not any(exp.lower() in feat.lower() for feat in detected_features):
                        missing.append(exp)
                
                if missing:
                    print(f"⚠️  Missing: {', '.join(missing)}")
                    results.append(False)
                else:
                    print("✅ All expected features found!")
                    results.append(True)
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append(False)
            
            print()
        
        # Summary
        successful = sum(results)
        total = len(results)
        
        print("=" * 80)
        print("📊 TEMPLATE GENERATION TEST SUMMARY")
        print("=" * 80)
        print(f"✅ Successful: {successful}/{total} ({(successful/total)*100:.1f}%)")
        
        if successful == total:
            print("🎉 All template generation tests passed!")
            print("🎯 Dynamic SQL generation is working correctly!")
        else:
            print("⚠️ Some template tests failed. Check output above.")
            
        return successful == total
        
    except Exception as e:
        print(f"❌ Template testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_template_generation())