"""
Check what metadata is stored in Pinecone for TimePeriod column
"""
import asyncio
from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore

async def check_pinecone():
    store = PineconeSchemaVectorStore()

    # Query for Nrx_SampleSummary table
    results = await store.search_relevant_tables("Reporting_BI_Nrx_SampleSummary TimePeriod", top_k=3)

    print("=" * 80)
    print("PINECONE METADATA FOR NRX_SAMPLESUMMARY")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\n📦 Result {i}:")
        print(f"   Table: {result.get('table_name')}")
        print(f"   Score: {result.get('score', 0):.4f}")
        
        metadata = result.get('metadata', {})
        print(f"   Metadata keys: {list(metadata.keys())}")
        
        # Check if TimePeriod column info exists
        columns = metadata.get('columns', [])
        print(f"   Total columns: {len(columns)}")
        
        # Find TimePeriod column
        for col in columns:
            if isinstance(col, dict) and 'TimePeriod' in col.get('column_name', ''):
                print(f"\n   🎯 Found TimePeriod column:")
                print(f"      Column data: {col}")
                break
        
        # Check column comments
        column_comments = metadata.get('column_comments', {})
        if 'TimePeriod' in column_comments:
            print(f"\n   💬 TimePeriod comment: {column_comments['TimePeriod']}")
        
        # Check if there are sample values
        if 'sample_values' in metadata:
            print(f"\n   📊 Sample values: {metadata.get('sample_values')}")

if __name__ == "__main__":
    asyncio.run(check_pinecone())
