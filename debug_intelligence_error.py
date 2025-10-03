#!/usr/bin/env python3
"""
Debug Intelligence Pipeline Error
=================================

This script isolates and debugs the 'str' object has no attribute 'get' error
in the intelligence pipeline.
"""

import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from query_intelligence.schema_analyzer import SchemaSemanticAnalyzer
from query_intelligence.intelligent_query_planner import IntelligentQueryPlanner

async def debug_intelligence_error():
    """Test the intelligence pipeline components individually"""
    
    print("🔍 DEBUGGING INTELLIGENCE PIPELINE ERROR")
    print("=" * 50)
    
    # Test 1: SchemaSemanticAnalyzer
    print("\n1️⃣ Testing SchemaSemanticAnalyzer...")
    try:
        analyzer = SchemaSemanticAnalyzer()
        print("✅ SchemaSemanticAnalyzer created successfully")
        
        # Test with simple metadata
        sample_metadata = {
            'Reporting_BI_PrescriberOverview': {
                'columns': [
                    {'name': 'PrescriberId', 'data_type': 'VARCHAR'},
                    {'name': 'PrescriberName', 'data_type': 'VARCHAR'},
                    {'name': 'RegionName', 'data_type': 'VARCHAR'}
                ]
            }
        }
        
        print(f"🔍 Testing analyze_schema_semantics with sample data...")
        result = await analyzer.analyze_schema_semantics(sample_metadata)
        print(f"✅ analyze_schema_semantics returned: {type(result)}")
        print(f"🔍 Result keys: {list(result.keys()) if isinstance(result, dict) else 'NOT A DICT'}")
        
        if isinstance(result, dict) and 'tables' in result:
            print("✅ analyze_schema_semantics working correctly")
        else:
            print("❌ analyze_schema_semantics not returning expected structure")
            
    except Exception as e:
        print(f"❌ SchemaSemanticAnalyzer error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: IntelligentQueryPlanner
    print("\n2️⃣ Testing IntelligentQueryPlanner initialization...")
    try:
        planner = IntelligentQueryPlanner()
        print("✅ IntelligentQueryPlanner created successfully")
        print(f"🔍 Schema analyzer type: {type(planner.schema_analyzer)}")
        
    except Exception as e:
        print(f"❌ IntelligentQueryPlanner error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: _format_semantic_analysis_for_llm method directly
    print("\n3️⃣ Testing _format_semantic_analysis_for_llm method...")
    try:
        planner = IntelligentQueryPlanner()
        
        # Create a sample semantic analysis result that should work
        sample_analysis = {
            'tables': {
                'Reporting_BI_PrescriberOverview': {
                    'columns': [
                        {'name': 'PrescriberId', 'data_type': 'VARCHAR', 'semantic_type': 'identifier'},
                        {'name': 'PrescriberName', 'data_type': 'VARCHAR', 'semantic_type': 'name'}
                    ],
                    'table_semantics': {
                        'primary_domain': 'healthcare',
                        'business_entities': ['prescriber']
                    }
                }
            },
            'cross_table_relationships': {},
            'business_domains': {}
        }
        
        print("🔍 Testing _format_semantic_analysis_for_llm with valid data...")
        result = planner._format_semantic_analysis_for_llm(sample_analysis, "Show me 5 prescribers")
        print(f"✅ _format_semantic_analysis_for_llm returned: {type(result)}")
        print(f"🔍 Result length: {len(result)} characters")
        
    except Exception as e:
        print(f"❌ _format_semantic_analysis_for_llm error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Check what happens when invalid data is passed
    print("\n4️⃣ Testing with invalid data types...")
    try:
        planner = IntelligentQueryPlanner()
        
        # Test with string instead of dict (this might be the issue)
        invalid_analysis = "This is a string, not a dict"
        
        print("🔍 Testing _format_semantic_analysis_for_llm with invalid (string) data...")
        result = planner._format_semantic_analysis_for_llm(invalid_analysis, "Show me 5 prescribers")
        print(f"✅ _format_semantic_analysis_for_llm handled invalid data: {type(result)}")
        
    except Exception as e:
        print(f"❌ _format_semantic_analysis_for_llm failed to handle invalid data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_intelligence_error())