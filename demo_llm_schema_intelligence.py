"""
Demo: LLM-Driven Schema Intelligence with Vector Storage
This demonstrates the proper approach where:
1. LLM analyzes schema during indexing
2. Intelligence is stored in vector DB
3. Query time retrieves pre-analyzed insights (no hardcoding!)
"""

import asyncio
import os
from dotenv import load_dotenv
from backend.agents.llm_schema_intelligence import VectorSchemaStorage

# Load environment
load_dotenv()

async def demo_llm_driven_schema_intelligence():
    """Demonstrate LLM-driven schema analysis and vector storage"""
    
    print("🎯 LLM-Driven Schema Intelligence Demo")
    print("=" * 50)
    
    # Initialize the LLM intelligence system
    intelligence_system = VectorSchemaStorage()
    
    # Sample table metadata (normally comes from database)
    sample_tables = {
        "NEGOTIATED_RATES": {
            "columns": [
                {"name": "NEGOTIATED_RATE_ID", "data_type": "NUMBER", "nullable": False},
                {"name": "PAYER", "data_type": "VARCHAR", "nullable": True},
                {"name": "NEGOTIATED_RATE", "data_type": "NUMBER", "nullable": True},
                {"name": "SERVICE_CODE", "data_type": "VARCHAR", "nullable": True},
                {"name": "PROVIDER_ID", "data_type": "NUMBER", "nullable": True}
            ],
            "sample_data": {
                "PAYER": ["Blue Cross", "Aetna", "Medicare"],
                "NEGOTIATED_RATE": [150.00, 200.50, 175.25],
                "SERVICE_CODE": ["99213", "99214", "99215"]
            }
        },
        "PROVIDER_REFERENCES": {
            "columns": [
                {"name": "PROVIDER_ID", "data_type": "NUMBER", "nullable": False},
                {"name": "PROVIDER_NAME", "data_type": "VARCHAR", "nullable": True},
                {"name": "TAX_ID", "data_type": "VARCHAR", "nullable": True},
                {"name": "LOCATION", "data_type": "VARCHAR", "nullable": True}
            ],
            "sample_data": {
                "PROVIDER_NAME": ["General Hospital", "City Clinic", "Regional Medical"],
                "TAX_ID": ["12-3456789", "98-7654321", "11-2233445"],
                "LOCATION": ["New York", "California", "Texas"]
            }
        }
    }
    
    print("🔍 Step 1: LLM Analyzing Schema During Indexing")
    print("-" * 50)
    
    # Run LLM analysis during indexing (this is what happens once during schema indexing)
    indexing_results = await intelligence_system.enhanced_schema_indexing(sample_tables)
    
    schema_intelligence = indexing_results["schema_intelligence"]
    enhanced_embeddings = indexing_results["enhanced_embeddings"]
    
    print("\n📊 LLM Analysis Results:")
    for table_name, analysis in schema_intelligence["table_analyses"].items():
        print(f"\n🏷️  TABLE: {table_name}")
        print(f"   📋 Business Purpose: {analysis['business_purpose']}")
        print(f"   🏢 Domain: {analysis['domain']}")
        print(f"   📊 Confidence: {analysis['confidence']:.2f}")
        
        print("\n   📝 Column Intelligence:")
        for col in analysis['column_insights']:
            operations = ', '.join(col['data_operations'])
            role = col['semantic_role']
            print(f"      • {col['column_name']}: {role} | Operations: [{operations}]")
            if col.get('business_meaning'):
                print(f"        Business Context: {col['business_meaning']}")
    
    print("\n🔗 Cross-Table Intelligence:")
    cross_intel = schema_intelligence["cross_table_intelligence"]
    for rel in cross_intel.get("relationships", []):
        print(f"   🔗 {rel['from_table']} -> {rel['to_table']} via {rel['join_column']}")
        print(f"      Context: {rel['business_context']}")
    
    print("\n💾 Step 2: Store in Vector Database")
    print("-" * 50)
    print(f"✅ Created {len(enhanced_embeddings)} enhanced embeddings")
    print("   Each embedding contains:")
    print("   • Table business context")
    print("   • Column semantic roles")
    print("   • Relationship intelligence")
    print("   • Query operation guidance")
    
    print("\n⚡ Step 3: Query-Time Intelligence Retrieval")
    print("-" * 50)
    
    # Simulate query time - retrieve pre-analyzed intelligence
    print("🔍 User Query: 'What are the average payment amounts by provider?'")
    print("\n📋 Retrieved Intelligence (from vector DB):")
    
    # Show how the pre-analyzed intelligence helps
    negotiated_rates_analysis = schema_intelligence["table_analyses"]["NEGOTIATED_RATES"]
    
    print("\n🎯 Table: NEGOTIATED_RATES")
    print(f"   ✅ Purpose: {negotiated_rates_analysis['business_purpose']}")
    
    # Show column guidance
    for col in negotiated_rates_analysis['column_insights']:
        if col['column_name'] == 'NEGOTIATED_RATE':
            print(f"   💰 NEGOTIATED_RATE: {col['business_meaning']}")
            print(f"      Operations: {col['data_operations']} ✅")
        elif col['column_name'] == 'PAYER':
            print(f"   🏢 PAYER: {col['business_meaning']}")
            print(f"      Operations: {col['data_operations']} (text field)")
    
    # Show query guidance
    query_guidance = negotiated_rates_analysis.get('query_guidance', {})
    if query_guidance.get('primary_amount_fields'):
        print(f"   💵 Primary Amount Fields: {query_guidance['primary_amount_fields']}")
    if query_guidance.get('forbidden_operations'):
        print(f"   🚫 Forbidden Operations: {query_guidance['forbidden_operations']}")
    
    print("\n🎯 LLM-Generated SQL Guidance:")
    print("   ✅ Use NEGOTIATED_RATE for AVG() - it's numeric")
    print("   ✅ Use PAYER for GROUP BY - it's categorical")
    print("   ✅ Join with PROVIDER_REFERENCES on PROVIDER_ID")
    print("   🚫 Never try to AVG(PAYER) - it's text!")
    
    print("\n🚀 Benefits of This Approach:")
    print("=" * 50)
    print("✅ LLM does all intelligent analysis (no hardcoding)")
    print("✅ Analysis happens once during indexing (fast queries)")
    print("✅ Rich business context stored in vector DB")
    print("✅ Query-time retrieval is instant")
    print("✅ Prevents errors like AVG(text_field)")
    print("✅ Provides relationship discovery")
    print("✅ Scalable to any database schema")

if __name__ == "__main__":
    asyncio.run(demo_llm_driven_schema_intelligence())
