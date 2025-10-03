#!/usr/bin/env python3

import sys
import os
import asyncio
sys.path.append('backend')

from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator

def test_join_query():
    """Test if the territory underperformance query generates JOINs"""
    
    query = "I need to understand why some territories are underperforming - show me territories where we have good rep coverage and are doing lots of activities, but prescribers still aren't writing many prescriptions for our targeted products. Include the rep names so I can follow up."
    
    print(f"🧪 Testing query: {query[:100]}...")
    
    try:
        # Initialize orchestrator
        orchestrator = DynamicAgentOrchestrator()
        
        # Process the query (async)
        async def run_query():
            return await orchestrator.process_query(
                query,
                user_id='test_user',
                session_id='test_session'
            )
        
        response = asyncio.run(run_query())
        
        # Extract SQL
        sql = None
        if '2_query_generation' in response:
            task_result = response['2_query_generation']
            sql = task_result.get('sql_query', task_result.get('sql', ''))
        
        if sql:
            print(f"✅ Generated SQL ({len(sql)} chars):")
            print("=" * 60)
            print(sql)
            print("=" * 60)
            
            # Check for JOINs
            sql_upper = sql.upper()
            has_join = any(join_type in sql_upper for join_type in ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN'])
            
            print(f"\n🔗 Contains JOIN: {'✅ YES' if has_join else '❌ NO'}")
            
            if has_join:
                join_count = sql_upper.count('JOIN')
                print(f"📊 Number of JOINs: {join_count}")
                
                # Find specific join types
                join_types = []
                if 'INNER JOIN' in sql_upper: join_types.append('INNER')
                if 'LEFT JOIN' in sql_upper: join_types.append('LEFT')
                if 'RIGHT JOIN' in sql_upper: join_types.append('RIGHT')
                if 'FULL JOIN' in sql_upper: join_types.append('FULL')
                if ' JOIN ' in sql_upper and not join_types: join_types.append('IMPLICIT')
                
                print(f"🔧 JOIN Types: {', '.join(join_types)}")
            
            # Check for multiple tables
            table_indicators = ['FROM', 'JOIN']
            tables_mentioned = []
            for line in sql.split('\n'):
                for indicator in table_indicators:
                    if indicator in line.upper():
                        # Extract table name after indicator
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if indicator in part.upper() and i + 1 < len(parts):
                                table_name = parts[i + 1].strip('[](),')
                                if table_name not in tables_mentioned:
                                    tables_mentioned.append(table_name)
            
            print(f"📋 Tables involved: {len(tables_mentioned)}")
            for table in tables_mentioned[:5]:  # Show first 5
                print(f"   • {table}")
            
        else:
            print("❌ No SQL generated")
            print("Response keys:", list(response.keys()) if response else "No response")
        
        return has_join if sql else False
        
    except Exception as e:
        print(f"❌ Error testing query: {e}")
        return False

if __name__ == "__main__":
    test_join_query()