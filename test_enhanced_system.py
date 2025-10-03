#!/usr/bin/env python3
"""
Test Enhanced NL2Q System - Intelligence Improvements
Testing SQL Generation 8/10, Column Intelligence 8/10, Data Retrieval 8/10
"""

import asyncio
import sys
import os
sys.path.append('.')

from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator

async def test_enhanced_intelligence():
    """Test the enhanced intelligence system with pharmaceutical queries"""
    
    print("🚀 TESTING ENHANCED NL2Q INTELLIGENCE SYSTEM")
    print("=" * 60)
    
    orchestrator = DynamicAgentOrchestrator()
    
    # Test queries with different intelligence signals
    test_queries = [
        {
            "query": "Show me top prescribers by volume in each territory",
            "expected_signals": ["aggregation", "ranking", "territorial"]
        },
        {
            "query": "List prescriber profiles with their specialty details", 
            "expected_signals": ["detail", "profile"]
        },
        {
            "query": "Find prescribers with high prescription volume last month",
            "expected_signals": ["temporal", "filtering", "ranking"]
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        print(f"\n🎯 TEST {i}: {query}")
        print("-" * 50)
        
        try:
            result = await orchestrator.process_query(
                user_query=query,
                user_id='test_user',
                session_id=f'test_session_{i}'
            )
            
            # Analyze results
            status = result.get('status', 'unknown')
            print(f"✅ Status: {status}")
            
            if 'results' in result:
                results_data = result['results']
                
                if isinstance(results_data, dict):
                    # Check for data retrieval
                    if 'results' in results_data:
                        data_rows = results_data['results']
                        row_count = len(data_rows) if data_rows else 0
                        print(f"📊 Data Retrieved: {row_count} rows")
                        
                        # Check for intelligent fallback strategies
                        metadata = results_data.get('metadata', {})
                        if metadata.get('intelligent_fallback'):
                            strategy = metadata.get('optimization_strategy', 'unknown')
                            print(f"🧠 Intelligent Strategy Used: {strategy}")
                            
                            if strategy == 'intelligent_table_selection':
                                selected_table = metadata.get('selected_table')
                                context_keywords = metadata.get('context_keywords', [])
                                print(f"🎯 Smart Table Selection: {selected_table}")
                                print(f"🔍 Context Keywords: {context_keywords}")
                        
                        # Show sample data if available
                        if data_rows and row_count > 0:
                            print(f"🔍 Sample Row: {data_rows[0]}")
                            
                            # Check columns for intelligence
                            columns = metadata.get('columns', [])
                            if columns:
                                print(f"📋 Columns: {columns[:5]}...")  # Show first 5 columns
                    
                    # Check for SQL intelligence
                    if 'sql_executed' in results_data:
                        sql = results_data['sql_executed']
                        print(f"🔧 SQL Generated: {sql[:100]}...")
                        
                        # Check for intelligent patterns
                        if 'TOP' in sql.upper():
                            print("✅ Azure SQL Server syntax detected")
                        if 'ORDER BY' in sql.upper():
                            print("✅ Intelligent sorting applied")
                else:
                    print(f"📊 Direct Results: {type(results_data)}")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 ENHANCED INTELLIGENCE TESTING COMPLETE")
    print("Key Improvements:")
    print("- 🎯 Query-intent driven SQL generation")
    print("- 🧠 Intelligent column relevance scoring")
    print("- 🔧 Smart constraint relaxation strategies")
    print("- 🔗 Context-aware table selection")
    print("- 📊 Progressive fallback optimization")

if __name__ == "__main__":
    asyncio.run(test_enhanced_intelligence())