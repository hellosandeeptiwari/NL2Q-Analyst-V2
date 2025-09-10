#!/usr/bin/env python3
"""
Demo script to show the SQL error correction process with LLM
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def demonstrate_sql_error_correction():
    """Show exactly what gets sent to the LLM for SQL error correction"""
    
    # Original user query
    original_query = "show me top 15 providers who get paid above book of business average"
    
    # Failed SQL with syntax error (example from your output)
    failing_sql = '''WITH PaymentLevels AS (
    SELECT
        "HEALTHCARE_PRICING_ANALYTICS_SAMPLE"."SAMPLES"."PROVIDER_REFERENCES"."PROVIDER_ID",
        AVG("HEALTHCARE_PRICING_ANALYTICS_SAMPLE"."SAMPLES"."PROVIDER_REFERENCES"."PAYER") AS AveragePayment
    FROM
        "HEALTHCARE_PRICING_ANALYTICS_SAMPLE"."SAMPLES"."PROVIDER_REFERENCES"
    GROUP BY
        "HEALTHCARE_PRICING_ANALYTICS_SAMPLE"."SAMPLES"."PROVIDER_REFERENCES"."PROVIDER_ID"
),
BusinessAverage AS (
    SELECT
        AVG(AveragePayment) AS BookOfBusinessAverage
    FROM
        PaymentLevels
),
RankedProviders AS (
    SELECT
        P."PROVIDER_ID",
        P.AveragePayment,
        B.BookOfBusinessAverage,
        P.AveragePayment - B.BookOfBusinessAverage AS PaymentDifference,
        RANK() OVER (ORDER BY P.AveragePayment - B.BookOfBusinessAverage DESC) AS Rank
    FROM
        PaymentLevels P
    CROSS JOIN
        BusinessAverage B
)

SELECT
    "HEALTHCARE_PRICING_ANALYTICS_SAMPLE"."SAMPLES"."RANKEDPROVIDERS"."PROVIDER_ID",
    "HEALTHCARE_PRICING_ANALYTICS_SAMPLE"."SAMPLES"."RANKEDPROVIDERS"."AveragePayment",
    "HEALTHCARE_PRICING_ANALYTICS_SAMPLE"."SAMPLES"."RANKEDPROVIDERS"."BookOfBusinessAverage",
    "HEALTHCARE_PRICING_ANALYTICS_SAMPLE"."SAMPLES"."RANKEDPROVIDERS"."PaymentDifference,
    "HEALTHCARE_PRICING_ANALYTICS_SAMPLE"."SAMPLES"."RANKEDPROVIDERS"."Rank"
FROM
    RankedProviders
WHERE
    Rank <= 15 OR Rank >= (SELECT COUNT(*) FROM RankedProviders) - 14
LIMIT 100;'''
    
    # Error message from Snowflake
    error_message = "near \".\": syntax error"
    
    # Database context
    db_name = os.getenv("SNOWFLAKE_DATABASE", "HEALTHCARE_PRICING_ANALYTICS_SAMPLE")
    schema_name = os.getenv("SNOWFLAKE_SCHEMA", "SAMPLES")
    
    # This is the EXACT correction prompt sent to the LLM
    correction_prompt = f"""You are a SQL expert. The following SQL query has an error that needs to be fixed.

**Original User Question:** {original_query}

**Failing SQL Query:**
```sql
{failing_sql}
```

**Error Message:** {error_message}

**Database Context:**
- Database: {db_name}
- Schema: {schema_name}
- Available tables: ALL_RATES, METRICS, NEGOTIATED_RATES, PROVIDER_REFERENCES, SERVICE_DEFINITIONS, VOLUME

**Common Snowflake Issues to Fix:**
1. Missing quotes in column/table references
2. Incorrect table aliasing in CTEs
3. Wrong database.schema.table format
4. Syntax errors in CTE definitions

Please provide ONLY the corrected SQL query without any explanation. The query should:
- Use proper Snowflake syntax
- Reference tables as "{db_name}"."{schema_name}"."TABLE_NAME"
- Fix any syntax errors
- Maintain the original intent of the query

**Corrected SQL:**"""

    print("=" * 80)
    print("🔧 EXACT LLM PROMPTS FOR SQL ERROR CORRECTION")
    print("=" * 80)
    
    print("\n❌ ORIGINAL FAILING SQL:")
    print("-" * 50)
    print(failing_sql)
    
    print("\n⚠️ ERROR MESSAGE:")
    print("-" * 50)
    print(f'"{error_message}"')
    
    print("\n🔧 CORRECTION PROMPT SENT TO LLM:")
    print("-" * 50)
    print(correction_prompt)
    
    print("\n🎯 WHAT THE LLM RECEIVES FOR CORRECTION:")
    print("-" * 50)
    print("Messages sent to OpenAI API:")
    print(f"Model: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
    print(f"Temperature: 0.1")
    print(f"Max Tokens: 1500")
    print("Messages:")
    print("1. Role: user")
    print(f"   Content: [Correction prompt shown above - {len(correction_prompt)} characters]")
    
    print("\n🔍 ERROR CORRECTION PROCESS:")
    print("-" * 50)
    print("1. SQL execution fails with syntax error")
    print("2. Error message captured from Snowflake")
    print("3. Failed SQL + error + context sent to LLM")
    print("4. LLM analyzes error and provides correction")
    print("5. Corrected SQL extracted and retried")
    
    print("\n🐛 IDENTIFIED ISSUES IN FAILING SQL:")
    print("-" * 50)
    print("1. Missing closing quote: 'PaymentDifference,' → 'PaymentDifference'")
    print('2. Invalid table reference: "RANKEDPROVIDERS" (CTE alias, not table)')
    print("3. Should reference CTE alias 'RankedProviders' directly")
    print("4. CTE aliases don't need database.schema qualification")
    
    print("\n✅ EXPECTED CORRECTION:")
    print("-" * 50)
    print("LLM should:")
    print("- Fix the missing quote in 'PaymentDifference'")
    print("- Use 'RankedProviders.PROVIDER_ID' instead of full qualification")
    print("- Remove database.schema references for CTE columns")
    print("- Maintain the original business logic")

if __name__ == "__main__":
    demonstrate_sql_error_correction()
