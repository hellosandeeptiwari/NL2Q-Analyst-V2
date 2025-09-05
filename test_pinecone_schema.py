from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
import json

import asyncio

async def test_pinecone_schema():
    print("Testing Pinecone schema extraction...\n")
    
    # Test Pinecone search directly
    vector_store = PineconeSchemaVectorStore()
    
    user_query = "show me all recommended messages for players"
    matches = await vector_store.search_relevant_tables(user_query, top_k=5)
    
    print(f"Query: {user_query}")
    print(f"Pinecone found {len(matches)} matches")
    
    for i, match in enumerate(matches[:3]):
        print(f"\nMatch {i+1}:")
        print(f"Match structure: {list(match.keys())}")
        print(f"Match data: {match}")
        break  # Just check first one to see structure
    
    # Test schema extraction from matches
    print("\n" + "="*50)
    print("Testing schema extraction from matches...\n")
    
    orchestrator = DynamicAgentOrchestrator()
    schema_info = orchestrator._extract_schema_from_pinecone_matches(matches)
    
    print(f"Extracted schema info:")
    print(json.dumps(schema_info, indent=2))

if __name__ == "__main__":
    asyncio.run(test_pinecone_schema())
