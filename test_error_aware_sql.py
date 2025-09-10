"""
Test Error-Aware SQL Generation with Retry Logic
Tests the enhanced system that feeds SQL errors back to LLM for correction
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.nl2sql.enhanced_generator import generate_sql_with_error_feedback
from backend.nl2sql.guardrails import GuardrailConfig

# Mock error context (what the system would pass after a failed SQL attempt)
def test_error_aware_sql_generation():
    print("🧪 Testing Error-Aware SQL Generation")
    print("=" * 60)
    
    # Simulate the GA provider cost query that was failing
    natural_language = "for GA state, which providers are most expensive relative to average?"
    
    # Schema information
    schema_data = {
        'PROVIDER_COSTS': [
            'PROVIDER_ID', 'PROVIDER_NAME', 'STATE', 'TOTAL_COST', 
            'PATIENT_COUNT', 'OVERALL_PCT_OF_AVG'
        ],
        'COST_METRICS': [
            'PROVIDER_ID', 'AVERAGE_COST', 'PERCENTILE_RANK', 'COST_CATEGORY'
        ]
    }
    
    # Simulate first attempt error (the actual Snowflake error we encountered)
    error_context = """
PREVIOUS SQL ERRORS TO FIX:
Attempt 1: SQL compilation error: invalid identifier 'PC."total_cost"' at line 7, position 4

The error occurred in a CTE where column aliases were inconsistently quoted.
Generate corrected Snowflake SQL that fixes these specific errors.
"""
    
    # Enhanced schema with error context (what the retry system would pass)
    enhanced_schema = {
        'tables': schema_data,
        'error_context': error_context,
        'database_type': 'snowflake',
        'retry_attempt': 2
    }
    
    # Test the error-aware generation
    constraints = GuardrailConfig(
        enable_write=False,
        allowed_schemas=['SAMPLES'],
        default_limit=100
    )
    
    print(f"📝 Query: {natural_language}")
    print(f"🔄 Simulating Retry Attempt: {enhanced_schema['retry_attempt']}")
    print(f"❌ Previous Error: invalid identifier 'PC.\"total_cost\"'")
    print()
    
    try:
        result = generate_sql_with_error_feedback(natural_language, enhanced_schema, constraints)
        
        print("🤖 LLM Error-Aware Response:")
        print(f"📋 Rationale: {result.rationale}")
        print(f"🎯 Confidence: {result.confidence_score:.1%}")
        print()
        print("🔧 Generated SQL:")
        print("-" * 40)
        print(result.sql)
        print("-" * 40)
        print()
        
        # Check if the SQL addresses the error
        sql_upper = result.sql.upper()
        
        print("🔍 Error Fix Analysis:")
        
        # Check for proper CTE quoting
        if 'WITH' in sql_upper and '"' in result.sql:
            print("✅ Uses quoted identifiers in CTE")
        elif 'WITH' in sql_upper:
            print("⚠️  CTE detected but may lack proper quoting")
        
        # Check for proper table aliases
        if '"TOTAL_COST"' in result.sql or '"total_cost"' in result.sql:
            print("✅ Uses properly quoted column aliases")
        elif 'total_cost' in result.sql.lower():
            print("⚠️  total_cost referenced but may need quotes")
        
        # Check for GA state filtering
        if "'GA'" in result.sql or "= 'GA'" in result.sql:
            print("✅ Includes GA state filtering")
        
        # Check for relative comparison
        if 'AVG' in sql_upper or 'AVERAGE' in sql_upper or 'PCT' in sql_upper:
            print("✅ Includes relative/average comparison logic")
        
        print()
        print("📊 Expected Improvements:")
        print("- Should fix column alias quoting issues")
        print("- Should use consistent Snowflake syntax")
        print("- Should maintain business logic for GA provider costs")
        
        return result.sql
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

if __name__ == "__main__":
    test_error_aware_sql_generation()
