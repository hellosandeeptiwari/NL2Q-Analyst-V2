from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
import asyncio
import json

async def test_end_to_end_sql_generation():
    print("Testing end-to-end SQL generation with schema context...\n")
    
    orchestrator = DynamicAgentOrchestrator()
    query = "input_id Expected_Value_avg Action_rank"  # Query that finds our table at position 1
    
    try:
        # Step 1: Schema discovery (what the orchestrator does)
        print("🔍 Step 1: Schema Discovery")
        schema_result = await orchestrator._execute_schema_discovery({
            'original_query': query,  # Changed from user_query to original_query
            'context': {}
        })
        
        if schema_result.get('status') == 'completed':  # Changed from 'success' to 'completed'
            available_tables = schema_result.get('discovered_tables', [])  # Changed key name
            pinecone_matches = schema_result.get('pinecone_matches', [])
            
            print(f"✅ Found {len(available_tables)} available tables")
            print(f"✅ Got {len(pinecone_matches)} Pinecone matches")
            
            # Check if our target table is found
            target_found = False
            for match in pinecone_matches:
                if match.get('table_name') == 'Final_NBA_Output_python_06042025':
                    target_found = True
                    print(f"🎯 Target table found with score: {match.get('total_score', 'N/A')}")
                    break
            
            if not target_found:
                print("❌ Target table not in Pinecone matches")
                print("Top matches:")
                for i, match in enumerate(pinecone_matches[:3]):
                    print(f"  {i+1}. {match.get('table_name')} (score: {match.get('total_score', 'N/A')})")
            
            # Step 2: SQL Generation with more debugging
            print(f"\n🔧 Step 2: SQL Generation")
            print(f"Available tables: {available_tables}")
            print(f"Number of Pinecone matches: {len(pinecone_matches)}")
            
            sql_result = await orchestrator._generate_database_aware_sql(
                query="show me all recommended messages",  # Use a real query
                available_tables=available_tables,
                pinecone_matches=pinecone_matches
            )
            
            if sql_result.get('status') == 'success':
                generated_sql = sql_result.get('sql', '')
                print(f"✅ Generated SQL:")
                print(f"```sql\n{generated_sql}\n```")
                
                # Check if the SQL uses correct column names
                if 'Recommended_Msg_Overall' in generated_sql:
                    print(f"🎯 SUCCESS: SQL uses correct column name 'Recommended_Msg_Overall'!")
                elif 'recommended_message' in generated_sql.lower():
                    print(f"❌ FAIL: SQL still uses incorrect column name")
                else:
                    print(f"⚠️ Column name check inconclusive")
                    
            else:
                print(f"❌ SQL generation failed: {sql_result.get('error', 'Unknown error')}")
                
        else:
            print(f"❌ Schema discovery failed: {schema_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_end_to_end_sql_generation())
