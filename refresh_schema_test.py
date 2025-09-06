import requests

def refresh_and_check_schema():
    print("🔄 Refreshing schema cache from Pinecone...\n")
    
    try:
        # Step 1: Refresh schema from Pinecone
        print("📡 Calling /refresh-schema endpoint...")
        response = requests.get("http://localhost:8000/refresh-schema")
        
        if response.status_code == 200:
            refresh_result = response.json()
            print(f"✅ Schema refresh completed:")
            print(f"   📊 Tables loaded: {refresh_result.get('tables_count', 0)}")
            print(f"   🏀 NBA tables: {refresh_result.get('nba_tables', [])}")
            print(f"   🎯 Target tables: {refresh_result.get('target_tables', [])}")
            
            sample_columns = refresh_result.get('sample_columns', {})
            for table, columns in sample_columns.items():
                print(f"   📋 {table}: {columns}")
        else:
            print(f"❌ Schema refresh failed: {response.status_code}")
            print(response.text)
            return
            
        # Step 2: Verify schema cache is populated
        print(f"\n🔍 Verifying schema cache...")
        schema_response = requests.get("http://localhost:8000/schema")
        
        if schema_response.status_code == 200:
            schema = schema_response.json()
            print(f"✅ Schema cache now contains {len(schema)} tables")
            
            # Check for target table and column
            target_tables = [name for name in schema.keys() if 'Final_NBA_Output' in name]
            if target_tables:
                sample_table = target_tables[0]
                columns = schema[sample_table]
                print(f"\n📋 Target table '{sample_table}' columns: {list(columns.keys())}")
                
                if 'Recommended_Msg_Overall' in columns:
                    print(f"✅ SUCCESS: Target column 'Recommended_Msg_Overall' found in schema cache!")
                else:
                    print(f"❌ Target column 'Recommended_Msg_Overall' not found")
            else:
                print(f"⚠️ No target tables found in schema cache")
                
        else:
            print(f"❌ Failed to verify schema: {schema_response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    refresh_and_check_schema()
