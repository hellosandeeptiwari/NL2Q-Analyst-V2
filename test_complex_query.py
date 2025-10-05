"""
Test complex analytical query through the fixed orchestrator
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import orchestrator
from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator

async def test_query():
    print('=' * 80)
    print('TESTING COMPLEX ANALYTICAL QUERY')
    print('=' * 80)
    
    query = "I need to understand why some territories are underperforming - show me territories where we have good rep coverage and are doing lots of activities, but prescribers still aren't writing many prescriptions for our targeted products. Include the rep names so I can follow up."
    
    print(f'\nQuery: {query}\n')
    
    # Initialize orchestrator
    orchestrator = DynamicAgentOrchestrator()
    
    print('Initializing orchestrator...')
    await orchestrator.initialize_on_startup()
    
    print('\nProcessing query...\n')
    
    # Process the query
    result = await orchestrator.process_query(
        user_query=query,
        user_id='test_user',
        session_id='test_session'
    )
    
    print('\n' + '=' * 80)
    print('QUERY RESULTS')
    print('=' * 80)
    
    # Extract results from nested structure
    results_data = result.get('results', {})
    
    # Check if execution task exists
    execution_result = None
    for key, value in results_data.items():
        if 'execution' in key.lower() and isinstance(value, dict):
            execution_result = value
            break
    
    if execution_result:
        status = execution_result.get('status', 'unknown')
        data = execution_result.get('results', [])
        sql_executed = execution_result.get('sql_executed', 'N/A')
        metadata = execution_result.get('metadata', {})
        warning = execution_result.get('warning')
        
        print(f'Status: {status}')
        print(f'\nSQL Executed:')
        print(sql_executed)
        print(f'\nRows returned: {len(data)}')
        
        # Show warning if query was modified
        if warning:
            print(f'\n{warning}')
        
        # Show fallback info
        if metadata.get('fallback_used'):
            print(f'\nWARNING - Fallback Applied:')
            print(f'  Reason: {metadata.get("fallback_reason", "Unknown")}')
            print(f'  Original SQL: {metadata.get("original_sql", "N/A")[:100]}...')
        
        if data and len(data) > 0:
            print(f'\nSample data (first 5 rows):')
            for i, row in enumerate(data[:5], 1):
                print(f'  Row {i}: {row}')
        
        # Show metadata
        if metadata:
            print(f'\nExecution metadata:')
            print(f'  - Execution time: {metadata.get("execution_time", "N/A")} seconds')
            print(f'  - Execution attempts: {execution_result.get("execution_attempt", 1)}')
            if metadata.get('columns'):
                print(f'  - Columns: {", ".join(metadata["columns"][:5])}{"..." if len(metadata["columns"]) > 5 else ""}')
    else:
        print(f'Query failed: {result.get("error", "Unknown error")}')
        print(f'\nAvailable result keys: {list(results_data.keys())}')
        
        # Try to extract any SQL that was generated
        for key, value in results_data.items():
            if isinstance(value, dict) and 'sql' in value:
                print(f'\nSQL was generated in {key}:')
                print(value.get('sql', 'N/A')[:200])
    
    print('\n' + '=' * 80)

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_query())
