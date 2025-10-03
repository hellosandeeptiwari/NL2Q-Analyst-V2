#!/usr/bin/env python3
"""
Quick test to verify error fixes
"""

import asyncio
import sys
sys.path.append('.')

from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator

async def test_fixes():
    print('🔧 Testing error fixes...')
    orchestrator = DynamicAgentOrchestrator()
    
    result = await orchestrator.process_query(
        user_query='Show me 5 prescribers',
        user_id='test_user', 
        session_id='fix_test'
    )
    
    print(f'Status: {result.get("status")}')
    if 'results' in result and isinstance(result['results'], dict):
        sql_executed = result['results'].get('sql_executed', '')
        if sql_executed:
            print(f'SQL Generated: {sql_executed[:150]}...')
            if 'TOP' in sql_executed.upper():
                print('✅ Azure SQL Server syntax detected')
            if 'LIMIT' in sql_executed.upper():
                print('❌ Still using LIMIT syntax')
        
        metadata = result['results'].get('metadata', {})
        if metadata.get('azure_sql_fixed'):
            print('✅ Azure SQL fix applied')
        if metadata.get('intelligent_fallback'):
            print('✅ Intelligent fallback strategy used')

if __name__ == "__main__":
    asyncio.run(test_fixes())