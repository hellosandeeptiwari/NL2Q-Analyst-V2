#!/usr/bin/env python3
"""
Test Pinecone Schema Vector Search with Mock Data
Tests the vector search functionality without requiring Snowflake connection
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent / "backend"))

# Mock schema data that represents typical NBA tables
MOCK_SCHEMA_DATA = [
    {
        "table_name": "NBA_PERFORMANCE_OUTPUT",
        "table_schema": "ENHANCED_NBA",
        "columns": [
            {"column_name": "PLAYER_ID", "data_type": "NUMBER", "is_nullable": False},
            {"column_name": "PLAYER_NAME", "data_type": "VARCHAR", "is_nullable": False},
            {"column_name": "TEAM", "data_type": "VARCHAR", "is_nullable": True},
            {"column_name": "PERFORMANCE_SCORE", "data_type": "FLOAT", "is_nullable": True},
            {"column_name": "RECOMMENDATIONS", "data_type": "VARCHAR", "is_nullable": True},
            {"column_name": "AI_MESSAGE", "data_type": "VARCHAR", "is_nullable": True},
            {"column_name": "PROVIDER_INPUT", "data_type": "VARCHAR", "is_nullable": True},
            {"column_name": "LAST_UPDATED", "data_type": "TIMESTAMP", "is_nullable": True}
        ],
        "description": "NBA player performance analysis with AI recommendations"
    },
    {
        "table_name": "NBA_PLAYER_STATS",
        "table_schema": "ENHANCED_NBA", 
        "columns": [
            {"column_name": "PLAYER_ID", "data_type": "NUMBER", "is_nullable": False},
            {"column_name": "SEASON", "data_type": "VARCHAR", "is_nullable": False},
            {"column_name": "POINTS_PER_GAME", "data_type": "FLOAT", "is_nullable": True},
            {"column_name": "REBOUNDS_PER_GAME", "data_type": "FLOAT", "is_nullable": True},
            {"column_name": "ASSISTS_PER_GAME", "data_type": "FLOAT", "is_nullable": True}
        ],
        "description": "NBA player statistics by season"
    },
    {
        "table_name": "NBA_TEAM_ROSTER",
        "table_schema": "ENHANCED_NBA",
        "columns": [
            {"column_name": "TEAM_ID", "data_type": "NUMBER", "is_nullable": False},
            {"column_name": "TEAM_NAME", "data_type": "VARCHAR", "is_nullable": False},
            {"column_name": "PLAYER_ID", "data_type": "NUMBER", "is_nullable": False},
            {"column_name": "POSITION", "data_type": "VARCHAR", "is_nullable": True},
            {"column_name": "JERSEY_NUMBER", "data_type": "NUMBER", "is_nullable": True}
        ],
        "description": "NBA team rosters and player positions"
    },
    {
        "table_name": "NBA_GAME_RESULTS",
        "table_schema": "ENHANCED_NBA",
        "columns": [
            {"column_name": "GAME_ID", "data_type": "NUMBER", "is_nullable": False},
            {"column_name": "HOME_TEAM", "data_type": "VARCHAR", "is_nullable": False},
            {"column_name": "AWAY_TEAM", "data_type": "VARCHAR", "is_nullable": False},
            {"column_name": "HOME_SCORE", "data_type": "NUMBER", "is_nullable": True},
            {"column_name": "AWAY_SCORE", "data_type": "NUMBER", "is_nullable": True},
            {"column_name": "GAME_DATE", "data_type": "DATE", "is_nullable": True}
        ],
        "description": "NBA game results and scores"
    }
]

class MockResult:
    """Mock result object"""
    def __init__(self, rows, error=None):
        self.rows = rows
        self.error = error

class MockDBAdapter:
    """Mock database adapter for testing"""
    def get_schema_info(self):
        return MOCK_SCHEMA_DATA
    
    def run(self, query, dry_run=False):
        """Mock the run method to return expected table/column data"""
        if "SHOW TABLES" in query:
            # Return table names
            table_names = [table["table_name"] for table in MOCK_SCHEMA_DATA]
            rows = [[None, name] for name in table_names]  # Format: [schema, table_name]
            return MockResult(rows)
        elif "DESCRIBE TABLE" in query:
            # Extract table name from query
            table_name = query.split()[-1]
            # Find the table in mock data
            for table in MOCK_SCHEMA_DATA:
                if table["table_name"] == table_name:
                    # Return column information: [name, type, nullable]
                    rows = []
                    for col in table["columns"]:
                        nullable = "Y" if col["is_nullable"] else "N"
                        rows.append([col["column_name"], col["data_type"], nullable])
                    return MockResult(rows)
            return MockResult([], error=f"Table {table_name} not found")
        elif "SELECT COUNT" in query:
            # Return mock row count
            return MockResult([[100]])  # Mock 100 rows
        else:
            return MockResult([], error=f"Unknown query: {query}")

async def main():
    print("🚀 Testing Pinecone Vector Search with Mock NBA Schema Data...")
    
    from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
    
    try:
        pinecone_store = PineconeSchemaVectorStore()
        mock_adapter = MockDBAdapter()
        
        # Step 1: Index mock schema
        print("\n📝 Indexing mock NBA schema...")
        await pinecone_store.index_database_schema(mock_adapter)
        print("✅ Schema indexed successfully!")
        
        # Step 2: Test semantic search queries
        test_queries = [
            "nba output table with recommended message and provider input",
            "player statistics and performance data",
            "team roster information",
            "game scores and results"
        ]
        
        for query in test_queries:
            print(f"\n🔍 SEARCHING: '{query}'")
            results = await pinecone_store.search_relevant_tables(query, top_k=4)
            
            print(f"📊 TOP {len(results)} TABLE SUGGESTIONS:")
            for i, table in enumerate(results):
                print(f"{i+1}. {table['table_name']} (score: {table['best_score']:.3f})")
                print(f"   Types: {', '.join(table['chunk_types'])}")
                print(f"   Sample: {table['sample_content'][:80]}...")
            print("-" * 50)
        
        print("\n🎉 Pinecone vector search testing completed successfully!")
        print("✅ The system can now provide intelligent table suggestions")
        print("✅ Top 4 relevant tables will be suggested for each query")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
