"""
Check what's actually in the Pinecone index
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

def check_pinecone_contents():
    """Check Pinecone index contents"""
    
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "nl2q-schema-index")
    
    print("=" * 70)
    print("🔍 Checking Pinecone Index Contents")
    print("=" * 70)
    print(f"\nIndex Name: {index_name}")
    
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        
        # Get index stats
        stats = index.describe_index_stats()
        print(f"\n📊 Index Statistics:")
        print(f"   Total Vectors: {stats.total_vector_count}")
        print(f"   Dimension: {stats.dimension}")
        print(f"   Namespaces: {stats.namespaces}")
        
        if stats.total_vector_count == 0:
            print("\n⚠️ Index is EMPTY - No tables have been indexed yet!")
            return
        
        # Query to get sample vectors and see what tables are indexed
        print(f"\n🔍 Fetching sample vectors to identify indexed tables...")
        
        # Create a dummy query vector
        dummy_vector = [0.0] * stats.dimension
        
        # Query with no filter to get all tables
        results = index.query(
            vector=dummy_vector,
            top_k=100,  # Get many results
            include_metadata=True
        )
        
        # Extract unique table names
        table_names = set()
        database_names = set()
        
        for match in results.matches:
            metadata = match.metadata or {}
            if 'table_name' in metadata:
                table_names.add(metadata['table_name'])
            if 'database' in metadata:
                database_names.add(metadata['database'])
        
        print(f"\n✅ Found {len(table_names)} unique tables in index:")
        for table in sorted(table_names):
            print(f"   📊 {table}")
        
        if database_names:
            print(f"\n🗄️ Databases:")
            for db in sorted(database_names):
                print(f"   💾 {db}")
        
        # Show sample metadata
        if results.matches:
            print(f"\n📝 Sample metadata from first vector:")
            sample_metadata = results.matches[0].metadata
            for key, value in sample_metadata.items():
                value_str = str(value)[:100]  # Truncate long values
                print(f"   {key}: {value_str}")
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_pinecone_contents()
