from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
import asyncio

async def test_direct_table_search():
    print("Testing direct search for our target table...\n")
    
    vector_store = PineconeSchemaVectorStore()
    
    # Try various search terms
    searches = [
        "Final_NBA_Output_python_06042025",
        "06042025",
        "Final NBA Output 06042025",
        "input_id Expected_Value_avg Action_rank",
        "Recommended_Msg_Overall"
    ]
    
    target_table = "Final_NBA_Output_python_06042025"
    
    for search_query in searches:
        print(f"\n🔍 Search: '{search_query}'")
        matches = await vector_store.search_relevant_tables(search_query, top_k=10)
        
        found_position = None
        for i, match in enumerate(matches):
            if match['table_name'] == target_table:
                found_position = i + 1
                print(f"  🎯 FOUND at position {found_position} with score {match['total_score']:.3f}")
                break
        
        if not found_position:
            print(f"  ❌ Not found in top 10")
            print(f"     Top 3: {[m['table_name'] for m in matches[:3]]}")

if __name__ == "__main__":
    asyncio.run(test_direct_table_search())
