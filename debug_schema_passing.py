import requests
import json

def debug_schema_passing():
    print("🔍 Debugging schema passing to LLM...\n")
    
    # First, let's check what's in the schema cache
    print("=" * 60)
    print("STEP 1: Check schema cache contents")
    print("=" * 60)
    
    try:
        response = requests.get("http://localhost:8000/schema")
        if response.status_code == 200:
            schema = response.json()
            print(f"📊 Schema cache contains {len(schema)} tables")
            
            # Look for the tables that Pinecone found
            pinecone_tables = ['NBA_outputs', 'Final_NBA_Output_python_20250519', 'Final_NBA_Output_python_20250425']
            
            for table in pinecone_tables:
                if table in schema:
                    columns = schema[table]
                    print(f"✅ Found {table} in schema cache with {len(columns)} columns: {list(columns.keys())}")
                else:
                    print(f"❌ {table} NOT FOUND in schema cache")
                    
            # Look for any NBA-related tables in schema cache
            nba_tables = [name for name in schema.keys() if 'nba' in name.lower() or 'NBA' in name]
            print(f"\n🏀 NBA-related tables in schema cache: {nba_tables[:5]}...")
            
        else:
            print(f"❌ Failed to get schema cache: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error checking schema cache: {e}")
    
    print("\n" + "=" * 60)
    print("STEP 2: Test the LLM directly with correct schema")
    print("=" * 60)
    
    # Let's test the generate_sql function directly
    try:
        from backend.nl2sql.generator import generate_sql
        from backend.nl2sql.guardrails import GuardrailConfig
        
        # Create a small schema with the correct tables
        test_schema = {
            "NBA_outputs": {
                "input_id": "varchar",
                "Marketing_Action_Adj": "varchar", 
                "Recommended_Msg_Overall": "varchar",
                "Expected_Value_avg": "float",
                "Adjusted_Importance": "float",
                "Action_Effect": "float",
                "Action_rank": "int"
            },
            "Final_NBA_Output_python_20250519": {
                "input_id": "varchar",
                "Marketing_Action_Adj": "varchar",
                "Recommended_Msg_Overall": "varchar", 
                "Expected_Value_avg": "float",
                "Adjusted_Importance": "float",
                "Action_Effect": "float",
                "Action_rank": "int"
            }
        }
        
        print(f"🔧 Testing LLM with controlled schema containing {len(test_schema)} tables")
        for table, cols in test_schema.items():
            print(f"   📋 {table}: {list(cols.keys())}")
        
        guardrail_cfg = GuardrailConfig(
            enable_write=False,
            allowed_schemas=["public"],
            default_limit=100
        )
        
        query = "What are the recommended messages for NBA marketing actions?"
        
        generated = generate_sql(query, test_schema, guardrail_cfg)
        
        print(f"\n🔧 Generated SQL:")
        print("-" * 50)
        print(generated.sql)
        print("-" * 50)
        
        # Check if it uses the correct tables and columns
        if any(table in generated.sql for table in test_schema.keys()):
            print(f"✅ SUCCESS: SQL uses correct table from test schema!")
            
            if "Recommended_Msg_Overall" in generated.sql:
                print(f"🎯 PERFECT: SQL contains correct column 'Recommended_Msg_Overall'!")
            else:
                print(f"⚠️ PARTIAL: Uses correct table but missing target column")
        else:
            print(f"❌ FAILURE: SQL doesn't use any table from test schema")
            print(f"   Expected tables: {list(test_schema.keys())}")
        
    except Exception as e:
        print(f"❌ Direct LLM test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_schema_passing()
