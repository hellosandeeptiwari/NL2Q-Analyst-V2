"""
Debug script to see actual chunk content format
"""
import asyncio
from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore

async def debug_chunks():
    store = PineconeSchemaVectorStore()
    
    # Get raw Pinecone results
    dummy_vector = [0.0] * 3072
    results = store.index.query(
        vector=dummy_vector,
        filter={"table_name": {"$eq": "Reporting_BI_TerritoryPerformanceSummary"}},
        top_k=20,
        include_metadata=True
    )
    
    print("="*80)
    print("ACTUAL CHUNK CONTENT FROM PINECONE")
    print("="*80)
    
    for i, match in enumerate(results.matches, 1):
        chunk_type = match.metadata.get("chunk_type")
        content = match.metadata.get("content", "")
        
        print(f"\n{i}. CHUNK TYPE: {chunk_type}")
        print(f"   CONTENT LENGTH: {len(content)} chars")
        print(f"   CONTENT PREVIEW:")
        print("   " + "-"*76)
        # Show first 500 chars
        for line in content[:500].split('\n'):
            print(f"   {line}")
        print("   " + "-"*76)
        
        if chunk_type == "column_group":
            print("\n   ANALYZING COLUMN GROUP CONTENT:")
            lines = content.split('\n')
            for line in lines[:20]:
                if '•' in line or '-' in line:
                    print(f"   BULLET LINE: {line}")

if __name__ == "__main__":
    asyncio.run(debug_chunks())
