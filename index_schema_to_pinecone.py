#!/usr/bin/env python3
"""
Index Snowflake Schema into Pinecone Vector Store
This will chunk, embed, and store all 166 tables for semantic search
"""
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

async def index_schema():
    print("🚀 Starting Pinecone Schema Indexing for all 166 tables...")
    
    from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
    from backend.db.engine import get_adapter
    
    # Initialize
    pinecone_store = PineconeSchemaVectorStore()
    db_adapter = get_adapter("snowflake")
    
    print("✅ Pinecone and Snowflake initialized")
    
    # Index all tables from Snowflake
    print("📊 Starting schema indexing...")
    await pinecone_store.index_database_schema(db_adapter)
    
    print("🎉 Schema indexing complete!")
    print("✅ All tables chunked, embedded, and stored in Pinecone")
    print("✅ Ready for semantic search!")

if __name__ == "__main__":
    asyncio.run(index_schema())
