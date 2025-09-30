#!/usr/bin/env python3
"""
Debug SQL generation to identify the 'os' error
"""

def test_sql_generation():
    try:
        from backend.nl2sql.enhanced_generator import generate_sql, GuardrailConfig
        
        # Simple schema for testing
        schema = {
            'Reporting_BI_PrescriberProfile': [
                'RegionId', 'RegionName', 'TerritoryId', 'TerritoryName', 
                'PrescriberId', 'PrescriberName', 'ProductGroupName', 'TRX'
            ]
        }
        
        # Simple query
        query = "Show top 10 prescribers by TRX for Tirosint products"
        
        # Configure guardrails
        guardrails = GuardrailConfig(
            enable_write=False,
            allowed_schemas=['Reporting_BI_PrescriberProfile'],
            default_limit=1000
        )
        
        print("🔍 Testing SQL generation...")
        result = generate_sql(
            natural_language=query,
            schema_snapshot=schema,
            constraints=guardrails
        )
        
        print("✅ SQL Generation successful!")
        print(f"SQL: {result.sql}")
        print(f"Confidence: {result.confidence_score}")
        
    except Exception as e:
        print(f"❌ Error in SQL generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sql_generation()