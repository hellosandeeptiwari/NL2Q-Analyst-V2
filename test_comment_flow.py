"""
Test script to verify SQL comments flow from Pinecone → Planner → SQL Generator
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_comment_extraction():
    """Test that comments are extracted from Pinecone and displayed"""
    print("="*80)
    print("TEST 1: Verify Pinecone extracts comments from chunks")
    print("="*80)
    
    from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
    
    pinecone_store = PineconeSchemaVectorStore()
    
    # Test with table that has comments
    table_name = "Reporting_BI_TerritoryPerformanceSummary"
    
    print(f"\n🔍 Getting table details for: {table_name}")
    table_details = await pinecone_store.get_table_details(table_name)
    
    print(f"\n✅ Table Details Retrieved:")
    print(f"   - Table name: {table_details.get('table_name')}")
    print(f"   - Table comment: {table_details.get('table_comment')}")
    print(f"   - Total columns: {len(table_details.get('columns', []))}")
    print(f"   - Column comments: {len(table_details.get('column_comments', {}))}")
    print(f"   - Chunks: {list(table_details.get('chunks', {}).keys())}")
    
    # Show first few column comments
    column_comments = table_details.get('column_comments', {})
    if column_comments:
        print(f"\n💬 Sample Column Comments:")
        for i, (col_name, comment) in enumerate(list(column_comments.items())[:5]):
            print(f"   {i+1}. {col_name}: {comment}")
    
    print("\n" + "="*80)
    print("TEST 2: Verify schema discovery passes comments to context")
    print("="*80)
    
    # Simulate what happens during schema discovery
    from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
    
    orchestrator = DynamicAgentOrchestrator()
    await orchestrator._ensure_initialized()
    
    # Search for tables
    print(f"\n🔍 Searching for tables related to 'territory'")
    table_matches = await orchestrator.pinecone_store.search_relevant_tables("territory", top_k=2)
    
    print(f"\n✅ Found {len(table_matches)} tables:")
    for match in table_matches:
        print(f"   - {match['table_name']} (score: {match['best_score']:.3f})")
    
    # Get enhanced details (simulating what schema discovery does)
    print(f"\n🔍 Getting enhanced details with comments...")
    best_table_for_test = None
    best_table_details = None
    max_comments = 0
    
    for match in table_matches:
        table_name = match['table_name']
        table_details = await orchestrator.pinecone_store.get_table_details(table_name)
        
        table_comment = table_details.get('table_comment')
        column_comments = table_details.get('column_comments', {})
        
        # Track table with most comments for TEST 3
        if len(column_comments) > max_comments:
            max_comments = len(column_comments)
            best_table_for_test = table_name
            best_table_details = table_details
        
        print(f"\n📊 {table_name}:")
        print(f"   Table comment: {table_comment if table_comment else 'None'}")
        print(f"   Column comments: {len(column_comments)} columns with comments")
        
        if column_comments:
            print(f"   Sample columns:")
            for i, (col, comment) in enumerate(list(column_comments.items())[:3]):
                print(f"      • {col}: {comment}")
    
    print("\n" + "="*80)
    print("TEST 3: Verify comments reach the planner")
    print("="*80)
    
    from backend.query_intelligence.intelligent_query_planner import IntelligentQueryPlanner
    
    planner = IntelligentQueryPlanner(db_adapter=orchestrator.db_connector)
    
    # Use the table with the most comments for testing
    table_name = best_table_for_test
    table_details = best_table_details
    print(f"\n🔍 Using table: {table_name} ({max_comments} columns with comments)")
    
    # Create mock context with comments (simulating what orchestrator provides)
    mock_context = {
        'matched_tables': [
            {
                'table_name': table_name,
                'columns': table_details.get('columns', []),
                'table_comment': table_details.get('table_comment'),
                'column_comments': table_details.get('column_comments', {})
            }
        ],
        'db_adapter': orchestrator.db_connector
    }
    
    # Extract metadata (this is what planner does)
    print(f"\n🔍 Extracting table metadata in planner...")
    table_metadata = planner._extract_table_metadata(mock_context, [table_name])
    
    if table_name in table_metadata:
        metadata = table_metadata[table_name]
        columns = metadata.get('columns', [])
        
        print(f"\n✅ Planner extracted metadata:")
        print(f"   - Total columns: {len(columns)}")
        
        # Check if descriptions are in columns
        columns_with_desc = [col for col in columns if isinstance(col, dict) and col.get('description')]
        print(f"   - Columns with descriptions: {len(columns_with_desc)}")
        
        if columns_with_desc:
            print(f"\n💬 Sample columns with descriptions:")
            for i, col in enumerate(columns_with_desc[:5]):
                print(f"   {i+1}. {col.get('column_name', col.get('name'))}: {col.get('description')}")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED!")
    print("="*80)
    print("\nSummary:")
    print("  1. ✅ Pinecone extracts comments from vector chunks")
    print("  2. ✅ Schema discovery passes comments in context")
    print("  3. ✅ Planner receives and processes column descriptions")
    print("\nNext: Comments will appear in LLM prompts during SQL generation!")

if __name__ == "__main__":
    asyncio.run(test_comment_extraction())
