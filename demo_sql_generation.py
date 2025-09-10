#!/usr/bin/env python3
"""
Demo script to show the exact schema context and user query sent to LLM for SQL generation
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def demonstrate_sql_generation_prompt():
    """Show exactly what gets sent to the LLM for SQL generation"""
    
    # Simulated schema context (this is what comes from Pinecone vector search)
    table_schemas = [
        "Table: PROVIDER_REFERENCES\nColumns: PROVIDER_ID, PAYER, NEGOTIATED_RATE, PROVIDER_NAME, LOCATION",
        "Table: ALL_RATES\nColumns: BILLING_CODE, RATE_TYPE, AMOUNT, PROVIDER_ID, EFFECTIVE_DATE",
        "Table: NEGOTIATED_RATES\nColumns: RATE_ID, PROVIDER_ID, PAYER_ID, SERVICE_CODE, NEGOTIATED_AMOUNT",
        "Table: METRICS\nColumns: METRIC_ID, PROVIDER_ID, PERFORMANCE_SCORE, QUALITY_RATING, VOLUME_SCORE",
        "Table: SERVICE_DEFINITIONS\nColumns: SERVICE_CODE, DESCRIPTION, CATEGORY, BASE_RATE, COMPLEXITY_LEVEL",
        "Table: VOLUME\nColumns: PROVIDER_ID, SERVICE_CODE, VOLUME_COUNT, TIME_PERIOD, TOTAL_REVENUE"
    ]
    
    # Simulated user query
    user_query = "show me top 15 providers who get paid above book of business average"
    
    # Available tables discovered
    available_tables = ["PROVIDER_REFERENCES", "ALL_RATES", "NEGOTIATED_RATES"]
    
    # Database context from environment
    db_name = os.getenv("SNOWFLAKE_DATABASE", "HEALTHCARE_PRICING_ANALYTICS_SAMPLE")
    schema_name = os.getenv("SNOWFLAKE_SCHEMA", "SAMPLES")
    
    # Build schema context
    schema_context = "\n\n".join(table_schemas)
    schema_success_count = len([s for s in table_schemas if "AI vector discovery failed" not in s])
    
    # This is the EXACT system prompt sent to the LLM
    system_prompt = f"""You are an AI-powered SQL generator using vector-discovered database schemas.

Database Context:
- Engine: Snowflake
- Database: {db_name}  
- Schema: {schema_name}
- Full qualification: "{db_name}"."{schema_name}"."table_name"

AI-DISCOVERED SCHEMAS (Vector Embedding Success Rate: {(schema_success_count/len(available_tables)*100):.1f}%):
{schema_context}

AI SCHEMA RULES:
1. Use ONLY the column names discovered by AI vector embeddings above
2. Column names are extracted from AI embeddings and are case-sensitive
3. Always use double quotes: "{db_name}"."{schema_name}"."Table_Name"."Column_Name"
4. Trust the AI-discovered schema - these columns exist in the database
5. Use LIMIT 100 for safety

User Query: {user_query}

Generate executable Snowflake SQL using the AI-discovered schema above."""

    # This is the EXACT user prompt sent to the LLM
    user_prompt = f"""Generate a Snowflake SQL query for: {user_query}

Use these tables: {', '.join(available_tables)}

Return only the SQL query, properly formatted for Snowflake."""

    print("=" * 80)
    print("🤖 EXACT LLM PROMPTS FOR SQL GENERATION")
    print("=" * 80)
    
    print("\n📋 SYSTEM PROMPT:")
    print("-" * 50)
    print(system_prompt)
    
    print("\n👤 USER PROMPT:")
    print("-" * 50)
    print(user_prompt)
    
    print("\n🎯 WHAT THE LLM RECEIVES:")
    print("-" * 50)
    print("Messages sent to OpenAI API:")
    print(f"Model: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
    print(f"Temperature: 0.1")
    print(f"Max Tokens: 1500")
    print("Messages:")
    print("1. Role: system")
    print(f"   Content: [System prompt shown above - {len(system_prompt)} characters]")
    print("2. Role: user") 
    print(f"   Content: [User prompt shown above - {len(user_prompt)} characters]")
    
    print("\n🔍 SCHEMA DISCOVERY PROCESS:")
    print("-" * 50)
    print("1. User query analyzed by Pinecone vector search")
    print("2. Relevant tables identified from embeddings")
    print("3. Schema details extracted from vector matches")
    print("4. Schema context formatted for LLM consumption")
    print("5. LLM generates SQL using discovered schema")
    
    print("\n📊 CURRENT CONFIGURATION:")
    print("-" * 50)
    print(f"Database: {db_name}")
    print(f"Schema: {schema_name}")
    print(f"OpenAI Model: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
    print(f"Tables Available: {len(available_tables)}")
    print(f"Schema Success Rate: {(schema_success_count/len(available_tables)*100):.1f}%")

if __name__ == "__main__":
    demonstrate_sql_generation_prompt()
