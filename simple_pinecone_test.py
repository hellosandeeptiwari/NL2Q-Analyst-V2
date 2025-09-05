from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
import asyncio
import json

async def simple_pinecone_test():
    print("Simple Pinecone test for NBA table...\n")
    
    vector_store = PineconeSchemaVectorStore()
    
    # Search specifically for NBA table
    queries = [
        "Final_NBA_Output_python_06042025",
        "NBA Output",
        "recommended message overall",
        "input_id"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        matches = await vector_store.search_relevant_tables(query, top_k=3)
        
        for i, match in enumerate(matches):
            print(f"  Match {i+1}: {match['table_name']} (score: {match['total_score']:.3f})")
            if match['table_name'] == 'Final_NBA_Output_python_06042025':
                print(f"  ✅ FOUND TARGET TABLE!")
                print(f"  Sample content: {match['sample_content']}")
                
                # Try to get detailed info
                try:
                    details = await vector_store.get_table_details(match['table_name'])
                    print(f"  Details keys: {list(details.keys())}")
                    if 'chunks' in details:
                        print(f"  Chunk types: {list(details['chunks'].keys())}")
                except Exception as e:
                    print(f"  Error getting details: {e}")
                
                break

if __name__ == "__main__":
    asyncio.run(simple_pinecone_test())
