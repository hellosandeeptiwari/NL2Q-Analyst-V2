import requests

def check_schema_cache():
    print("🔍 Checking current schema cache...\n")
    
    try:
        response = requests.get("http://localhost:8000/schema")
        
        if response.status_code == 200:
            schema = response.json()
            print(f"📊 Schema cache contains {len(schema)} tables:")
            
            for table_name, columns in list(schema.items())[:5]:
                print(f"   📋 {table_name}: {list(columns.keys())}")
            
            if len(schema) > 5:
                print(f"   ... and {len(schema) - 5} more tables")
                
            # Check if NBA tables are in schema
            nba_tables = [name for name in schema.keys() if 'nba' in name.lower() or 'NBA' in name]
            print(f"\n🏀 NBA-related tables in schema: {nba_tables}")
            
            # Check for our target table
            target_tables = [name for name in schema.keys() if 'Final_NBA_Output' in name]
            print(f"🎯 Target NBA output tables: {target_tables}")
            
            if target_tables:
                sample_table = target_tables[0]
                columns = schema[sample_table]
                print(f"\n📋 Sample table '{sample_table}' columns:")
                for col, datatype in columns.items():
                    print(f"   - {col}: {datatype}")
                    
                # Check for target column
                if 'Recommended_Msg_Overall' in columns:
                    print(f"✅ Found target column 'Recommended_Msg_Overall'!")
                else:
                    print(f"❌ Target column 'Recommended_Msg_Overall' not found")
                    print(f"   Available columns: {list(columns.keys())}")
            
        else:
            print(f"❌ Failed to get schema: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error checking schema: {e}")

if __name__ == "__main__":
    check_schema_cache()
