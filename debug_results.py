#!/usr/bin/env python3
"""
Debug script to check the actual API response structure
"""

import json
import sys
import os

# Add the backend directory to Python path
sys.path.append('backend')

# Mock a simple execution result to see the expected structure
def debug_execution_result():
    """Debug what the execution step should return"""
    
    # This simulates what _execute_query_execution returns
    mock_execution_result = {
        "results": [
            {"NPI": "1234567890", "PROVIDER_NAME": "Test Provider", "Share": 0.85},
            {"NPI": "2345678901", "PROVIDER_NAME": "Another Provider", "Share": 0.92},
            {"NPI": "3456789012", "PROVIDER_NAME": "Third Provider", "Share": 0.78}
        ],
        "row_count": 3,
        "execution_time": 0.5,
        "metadata": {
            "columns": ["NPI", "PROVIDER_NAME", "Share"],
            "was_sampled": False,
            "job_id": "test_job_123"
        },
        "sql_executed": "SELECT NPI, PROVIDER_NAME, Share FROM test_table LIMIT 10",
        "execution_attempt": 1,
        "status": "completed"
    }
    
    # This simulates the overall orchestrator response
    mock_orchestrator_response = {
        "plan_id": "plan_test_123",
        "user_query": "list providers whose share within their NPI is unusually high",
        "reasoning_steps": ["Planned 4 execution steps", "Analyzed database structure", "Coordinated query processing"],
        "estimated_execution_time": "8s",
        "tasks": [
            {"task_type": "schema_discovery", "agent": "dynamic"},
            {"task_type": "semantic_understanding", "agent": "dynamic"},
            {"task_type": "query_generation", "agent": "dynamic"},
            {"task_type": "execution", "agent": "dynamic"}
        ],
        "status": "completed",
        "results": {
            "1_schema_discovery": {"status": "completed", "tables_found": ["PROVIDER_REFERENCES", "VOLUME", "METRICS"]},
            "2_semantic_understanding": {"status": "completed"},
            "3_query_generation": {"status": "completed", "sql_query": "SELECT * FROM test"},
            "4_execution": mock_execution_result  # This is the key part!
        }
    }
    
    print("🔍 Expected API Response Structure:")
    print(json.dumps(mock_orchestrator_response, indent=2))
    
    print("\n📊 Frontend should find:")
    print(f"- Execution step key: '4_execution'")
    print(f"- Has results property: {bool(mock_orchestrator_response['results']['4_execution']['results'])}")
    print(f"- Results count: {len(mock_orchestrator_response['results']['4_execution']['results'])}")
    print(f"- Status: {mock_orchestrator_response['results']['4_execution']['status']}")
    
    return mock_orchestrator_response

if __name__ == "__main__":
    debug_execution_result()
