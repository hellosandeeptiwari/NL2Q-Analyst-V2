#!/usr/bin/env python3
"""
Script to create a new Pinecone index with the correct dimensions for text-embedding-3-large
"""

import os
import pinecone
from dotenv import load_dotenv

def create_new_pinecone_index():
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('PINECONE_API_KEY')
    environment = os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp')
    index_name = os.getenv('PINECONE_INDEX_NAME', 'nl2q-schema-index')
    
    if not api_key:
        print("❌ PINECONE_API_KEY not found in environment variables")
        return
    
    # Initialize Pinecone
    pinecone.init(api_key=api_key, environment=environment)
    
    # Check if index already exists
    existing_indexes = pinecone.list_indexes()
    print(f"📋 Existing indexes: {existing_indexes}")
    
    # Create a new index name with correct dimensions
    new_index_name = f"{index_name}-3072"
    
    if new_index_name in existing_indexes:
        print(f"✅ Index '{new_index_name}' already exists with correct dimensions")
        print(f"💡 Update your .env file to use: PINECONE_INDEX_NAME={new_index_name}")
        return
    
    print(f"🚀 Creating new index '{new_index_name}' with 3072 dimensions...")
    
    try:
        pinecone.create_index(
            name=new_index_name,
            dimension=3072,  # For text-embedding-3-large
            metric='cosine',
            pods=1,
            replicas=1,
            pod_type='p1.x1'
        )
        
        print(f"✅ Successfully created index '{new_index_name}'")
        print(f"💡 Update your .env file to use: PINECONE_INDEX_NAME={new_index_name}")
        print(f"📝 Or keep the current embedding model as text-embedding-3-small")
        
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        print(f"💡 You may need to delete the existing index first or use a different name")

if __name__ == "__main__":
    create_new_pinecone_index()
