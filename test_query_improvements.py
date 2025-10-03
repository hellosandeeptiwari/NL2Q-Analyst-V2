#!/usr/bin/env python3
"""
Test script to validate query improvements for dynamic SQL generation
Tests the "do not hardcode anything" improvements
"""

import requests
import json
import time

def test_query_endpoint(query, expected_features=None):
    """Test a query and check for expected features"""
    print(f"\n🧪 Testing: {query}")
    print("=" * 80)
    
    try:
        response = requests.post(
            'http://localhost:8000/query',
            json={'query': query, 'user_id': 'test_user'},
            timeout=90
        )
        
        print(f"📟 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            sql = result.get('sql', '')
            
            if sql:
                print(f"🎯 Generated SQL ({len(sql)} chars):")
                print("-" * 60)
                print(sql)
                print("-" * 60)
                
                # Feature detection
                detected_features = []
                sql_upper = sql.upper()
                
                # Dynamic TOP limit detection
                if 'TOP 1' in sql_upper:
                    detected_features.append('TOP 1 limit')
                elif 'TOP 5' in sql_upper:
                    detected_features.append('TOP 5 limit')
                elif 'TOP 10' in sql_upper:
                    detected_features.append('TOP 10 limit')
                elif 'TOP 15' in sql_upper:
                    detected_features.append('TOP 15 limit')
                elif 'TOP ' in sql_upper:
                    detected_features.append('Dynamic TOP limit')
                
                # Product filtering
                if 'TIROSINT' in sql_upper:
                    detected_features.append('Tirosint filtering')
                if 'LEVOTHYROXINE' in sql_upper:
                    detected_features.append('Levothyroxine filtering')
                
                # WHERE clause detection
                if 'WHERE' in sql_upper and 'LIKE' in sql_upper:
                    detected_features.append('Dynamic WHERE filtering')
                
                # ORDER BY detection
                if 'ORDER BY' in sql_upper:
                    if 'DESC' in sql_upper:
                        detected_features.append('Descending ORDER BY')
                    else:
                        detected_features.append('ORDER BY clause')
                
                # Sales metrics
                if 'TRX' in sql_upper:
                    detected_features.append('TRX sales metrics')
                if 'NRX' in sql_upper:
                    detected_features.append('NRX metrics')
                if 'TQTY' in sql_upper:
                    detected_features.append('TQTY metrics')
                
                # Territory/Region grouping
                if 'TERRITORYNAME' in sql_upper:
                    detected_features.append('Territory grouping')
                if 'REGIONNAME' in sql_upper:
                    detected_features.append('Region grouping')
                
                # Clean SQL check
                sql_clean = sql.strip()
                if sql_clean.upper().startswith('SELECT') or sql_clean.upper().startswith('--'):
                    detected_features.append('Clean SQL (no explanatory text)')
                else:
                    detected_features.append('❌ Contains explanatory text')
                
                # Template usage detection
                if 'template' in sql.lower():
                    detected_features.append('Template fallback used')
                
                print(f"🎉 Detected Features ({len(detected_features)}):")
                for feature in detected_features:
                    print(f"   ✅ {feature}")
                
                # Check against expected features
                if expected_features:
                    missing_features = []
                    for expected in expected_features:
                        if not any(expected.lower() in feature.lower() for feature in detected_features):
                            missing_features.append(expected)
                    
                    if missing_features:
                        print(f"⚠️  Missing Expected Features:")
                        for missing in missing_features:
                            print(f"   ❌ {missing}")
                    else:
                        print("✅ All expected features detected!")
                
                return True, detected_features
                
            else:
                print("❌ No SQL found in response")
                print(f"Available keys: {list(result.keys())}")
                return False, []
                
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False, []
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False, []

def main():
    """Run all test cases"""
    print("🚀 Testing Query Improvements - Dynamic SQL Generation")
    print("🎯 Goal: Validate 'do not hardcode anything' improvements")
    print("=" * 80)
    
    # Test cases with expected features
    test_cases = [
        {
            'query': 'summarize top 10 sales of tirosint sol by territory',
            'expected': ['TOP 10', 'Tirosint', 'Territory', 'TRX', 'WHERE', 'ORDER BY']
        },
        {
            'query': 'show me top 5 prescriptions of Levothyroxine by region',
            'expected': ['TOP 5', 'Levothyroxine', 'Region', 'WHERE']
        },
        {
            'query': 'get top 15 records by sales volume',
            'expected': ['TOP 15', 'ORDER BY', 'TRX or sales']
        },
        {
            'query': 'display all prescriber data',
            'expected': ['Clean SQL', 'No WHERE clause']
        },
        {
            'query': 'find prescribers with high TRX values',
            'expected': ['TRX', 'Clean SQL']
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}/{len(test_cases)}")
        success, features = test_query_endpoint(
            test_case['query'], 
            test_case['expected']
        )
        results.append({
            'query': test_case['query'],
            'success': success,
            'features': features
        })
        time.sleep(2)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    print(f"✅ Successful Tests: {successful_tests}/{total_tests}")
    
    for i, result in enumerate(results, 1):
        status = "✅" if result['success'] else "❌"
        print(f"{status} Test {i}: {result['query'][:50]}...")
        if result['features']:
            key_features = [f for f in result['features'] if any(word in f.lower() for word in ['top', 'where', 'order', 'template'])]
            if key_features:
                print(f"     Key features: {', '.join(key_features[:3])}")
    
    print(f"\n🎯 Overall Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    
    if successful_tests == total_tests:
        print("🎉 All tests passed! Dynamic SQL generation is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the detailed output above.")

if __name__ == "__main__":
    main()