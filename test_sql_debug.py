#!/usr/bin/env python3
"""
Debug SQL execution issues
"""

from backend.db.engine import get_adapter
import traceback

def test_sql_connection():
    print('🔍 Testing SQL connection and query...')
    try:
        adapter = get_adapter('snowflake')
        print('✅ Database adapter created')
        
        # Test a simple query first
        simple_sql = 'SELECT COUNT(*) FROM "HEALTHCARE_PRICING_ANALYTICS_SAMPLE"."SAMPLES"."METRICS"'
        print(f'🔍 Testing simple query: {simple_sql[:50]}...')
        test_result = adapter.run(simple_sql)
        
        if test_result.error:
            print(f'❌ Simple query failed: {test_result.error}')
            return False
        else:
            print(f'✅ Simple query success: {test_result.rows[0][0]} rows in METRICS')
        
        # Test the corrected complex query
        complex_sql = '''
WITH AverageCosts AS (
    SELECT
        "NPI",
        AVG("OVERALL_PCT_OF_AVG") AS "avg_cost"
    FROM
        "HEALTHCARE_PRICING_ANALYTICS_SAMPLE"."SAMPLES"."METRICS"
    GROUP BY
        "NPI"
),
ProviderCosts AS (
    SELECT
        ar."NPI",
        ar."PROVIDER_GROUP_ID",
        ar."STATE",
        SUM(m."OVERALL_PCT_OF_AVG") AS "total_cost"        
    FROM
        "HEALTHCARE_PRICING_ANALYTICS_SAMPLE"."SAMPLES"."ALL_RATES" ar
    JOIN
        "HEALTHCARE_PRICING_ANALYTICS_SAMPLE"."SAMPLES"."METRICS" m
    ON
        ar."NPI" = m."NPI"
    WHERE
        ar."STATE" = 'GA'
    GROUP BY
        ar."NPI", ar."PROVIDER_GROUP_ID", ar."STATE"     
)
SELECT
    pc."NPI",
    pc."PROVIDER_GROUP_ID",
    pc."total_cost",
    ac."avg_cost",
    (pc."total_cost" / ac."avg_cost") AS "cost_relative_to_avg"
FROM
    ProviderCosts pc
JOIN
    AverageCosts ac
ON
    pc."NPI" = ac."NPI"
ORDER BY
    "cost_relative_to_avg" DESC
LIMIT 5
'''
        
        print('🔍 Testing complex query...')
        result = adapter.run(complex_sql)
        
        if result.error:
            print(f'❌ Complex query failed: {result.error}')
            print(f'❌ Error type: {type(result.error)}')
            
            # Try a simpler version to isolate the issue
            print('🔍 Testing JOIN without aggregation...')
            simple_join_sql = '''
SELECT 
    ar."NPI",
    ar."STATE",
    m."OVERALL_PCT_OF_AVG"
FROM 
    "HEALTHCARE_PRICING_ANALYTICS_SAMPLE"."SAMPLES"."ALL_RATES" ar
JOIN 
    "HEALTHCARE_PRICING_ANALYTICS_SAMPLE"."SAMPLES"."METRICS" m
ON 
    ar."NPI" = m."NPI"
WHERE 
    ar."STATE" = 'GA'
LIMIT 5
'''
            join_result = adapter.run(simple_join_sql)
            if join_result.error:
                print(f'❌ Simple JOIN failed: {join_result.error}')
            else:
                print(f'✅ Simple JOIN works: {len(join_result.rows)} rows')
                
        else:
            print(f'✅ Complex query success: {len(result.rows)} rows returned')
            if result.rows:
                print(f'Sample row: {result.rows[0]}')
                
    except Exception as e:
        print(f'❌ Connection error: {e}')
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    test_sql_connection()
