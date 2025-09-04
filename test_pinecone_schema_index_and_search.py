#!/usr/bin/env python3
"""
Test Pinecone Schema Indexing and Semantic Search
Indexes your Snowflake schema and runs a sample semantic search query
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent / "backend"))

async def main():
    print("🚀 Starting Pinecone schema indexing and search test...")
    from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
    from backend.db.engine import get_adapter

    pinecone_store = PineconeSchemaVectorStore()
    db_adapter = get_adapter("snowflake")

    # Step 1: Index schema
    print("\n📝 Indexing schema...")
    await pinecone_store.index_database_schema(db_adapter)

    # Step 2: Test semantic search
    print("\n🔍 Testing semantic search...")
    test_query = "nba output table with recommended message and provider input"
    results = await pinecone_store.search_relevant_tables(test_query, top_k=4)

    print(f"\n📊 TOP 4 TABLE SUGGESTIONS FOR QUERY: '{test_query}'")
    for i, table in enumerate(results):
        print(f"{i+1}. {table['table_name']} (score: {table['best_score']:.3f})")
        print(f"   Types: {', '.join(table['chunk_types'])}")
        print(f"   Sample: {table['sample_content'][:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
