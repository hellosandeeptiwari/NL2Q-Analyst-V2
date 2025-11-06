"""
Debug script to trace query planning for the Tirosint query
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test_query_planning():
    print("="*80)
    print("DEBUGGING QUERY PLANNING")
    print("="*80)
    
    from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
    
    # Initialize orchestrator
    print("\n🔧 Initializing orchestrator...")
    orchestrator = DynamicAgentOrchestrator()
    await orchestrator._ensure_initialized()
    
    # Test query
    user_query = "What was the total number of prescriptions (TRx) for Tirosint Capsules during Q2 2022?"
    
    print(f"\n📝 User Query: {user_query}")
    print("\n" + "="*80)
    print("STEP 1: Schema Discovery (Pinecone Search)")
    print("="*80)
    
    # Search for relevant tables
    table_matches = await orchestrator.pinecone_store.search_relevant_tables(user_query, top_k=3)
    
    print(f"\n✅ Found {len(table_matches)} tables:")
    for i, match in enumerate(table_matches, 1):
        print(f"\n{i}. {match['table_name']} (score: {match['best_score']:.3f})")
        
        # Get table details
        details = await orchestrator.pinecone_store.get_table_details(match['table_name'])
        columns = details.get('columns', [])
        column_comments = details.get('column_comments', {})
        
        print(f"   Columns: {len(columns)}")
        print(f"   Columns with comments: {len(column_comments)}")
        
        # Check for relevant columns
        relevant_cols = []
        for col_name in columns:
            col_lower = col_name.lower()
            if 'trx' in col_lower or 'prescription' in col_lower or 'product' in col_lower or 'tirosint' in col_lower:
                comment = column_comments.get(col_name, 'No comment')
                relevant_cols.append(f"{col_name} ({comment})")
        
        if relevant_cols:
            print(f"   📊 Relevant columns found:")
            for col in relevant_cols[:5]:
                print(f"      • {col}")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # Check which table should be selected
    print("\n🔍 Expected table: Reporting_BI_TerritoryPerformanceSummary")
    print("   Why? Has TRX column + ProductGroupName for filtering")
    
    print("\n🔍 Actual table selected: Reporting_BI_Nrx_SampleSummary")
    print("   Why was this chosen?")
    
    # Check if TerritoryPerformanceSummary is in results
    found_correct_table = False
    for match in table_matches:
        if match['table_name'] == 'Reporting_BI_TerritoryPerformanceSummary':
            found_correct_table = True
            print(f"\n   ✅ Correct table IS in Pinecone results (score: {match['best_score']:.3f})")
            
            # Check its columns
            details = await orchestrator.pinecone_store.get_table_details(match['table_name'])
            columns = details.get('columns', [])
            column_comments = details.get('column_comments', {})
            
            has_trx = 'TRX' in columns
            has_product = 'ProductGroupName' in columns
            
            print(f"   ✅ Has TRX column: {has_trx}")
            print(f"   ✅ Has ProductGroupName column: {has_product}")
            
            if has_trx:
                trx_comment = column_comments.get('TRX', 'No comment')
                print(f"   💬 TRX comment: {trx_comment}")
            
            if has_product:
                product_comment = column_comments.get('ProductGroupName', 'No comment')
                print(f"   💬 ProductGroupName comment: {product_comment}")
            
            break
    
    if not found_correct_table:
        print("\n   ❌ Correct table NOT in Pinecone results!")
        print("   Issue: Vector search not returning the right table")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print("\n1. Check vector embeddings:")
    print("   - Does 'Tirosint Capsules' match well with table descriptions?")
    print("   - Does 'total prescriptions' match with TRX comment?")
    
    print("\n2. Check planner logic:")
    print("   - Is it choosing the table with best semantic match?")
    print("   - Is it considering column availability (TRX, ProductGroupName)?")
    
    print("\n3. Check SQL generation:")
    print("   - Is it using column comments to understand TRX = prescriptions?")
    print("   - Is it understanding Q2 2022 = 'Q2 2022' format?")

if __name__ == "__main__":
    asyncio.run(test_query_planning())
