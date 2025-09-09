#!/usr/bin/env python3
"""
Script to create a new Pinecone index with the correct dimensions for text-embedding-3-large
"""

import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_new_pinecone_index():
    """Create a new Pinecone index with 3072 dimensions for text-embedding-3-large"""
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    
    # Configuration
    index_name = "nl2q-schema-index-3072"  # New index name
    dimension = 3072  # Dimension for text-embedding-3-large
    
    print(f"🚀 Creating new Pinecone index: {index_name}")
    print(f"📏 Dimensions: {dimension}")
    
    # Let's try different cloud/region combinations
    cloud_regions = [
        ('aws', 'us-east-1'),
        ('gcp', 'us-central1'),
        ('azure', 'eastus'),
        ('aws', 'us-west-2'),
        ('gcp', 'us-west1'),
    ]
    
    try:
        # Check if index already exists
        existing_indexes = pc.list_indexes()
        if index_name in [idx.name for idx in existing_indexes]:
            print(f"⚠️  Index {index_name} already exists!")
            response = input("Do you want to delete and recreate it? (y/N): ")
            if response.lower() == 'y':
                print(f"🗑️  Deleting existing index: {index_name}")
                pc.delete_index(index_name)
                time.sleep(10)  # Wait for deletion to complete
            else:
                print("❌ Aborted. Index already exists.")
                return False
        
        # Try different cloud/region combinations
        for cloud, region in cloud_regions:
            try:
                print(f"🔨 Trying to create index with cloud: {cloud}, region: {region}...")
                pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud=cloud,
                        region=region
                    )
                )
                
                print(f"✅ Index created successfully with {cloud}/{region}!")
                break
                
            except Exception as region_error:
                print(f"❌ Failed with {cloud}/{region}: {region_error}")
                continue
        else:
            print("❌ Failed to create index with any cloud/region combination")
            return False
        
        # Wait for index to be ready
        print("⏳ Waiting for index to be ready...")
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        
        print(f"✅ Index {index_name} created successfully!")
        print(f"\n📋 Next steps:")
        print(f"1. Update your .env file:")
        print(f"   PINECONE_INDEX_NAME={index_name}")
        print(f"   EMBED_MODEL=text-embedding-3-large")
        print(f"2. Restart your backend server")
        print(f"3. Run the indexing process")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Pinecone Index Creator for text-embedding-3-large")
    print("=" * 50)
    
    success = create_new_pinecone_index()
    
    if success:
        print("\n🎉 Index creation completed successfully!")
    else:
        print("\n💥 Index creation failed!")
