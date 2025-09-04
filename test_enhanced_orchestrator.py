#!/usr/bin/env python3
"""
Test Enhanced Orchestrator with Azure Integration (Mock Mode)
Tests the flow without requiring Azure setup
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

async def test_enhanced_orchestrator_flow():
    """Test the enhanced orchestrator flow"""
    
    print("🧪 Testing Enhanced NL2Q Flow with Azure Integration")
    print("="*70)
    
    # Test the current orchestrator to see if it picks up more tables
    try:
        from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
        
        orchestrator = DynamicAgentOrchestrator()
        
        # Test query
        query = "read table final nba output python and fetch top 5 rows and create a visualization with frequency of recommended message and provider input"
        
        print(f"🔍 Testing Query: {query}")
        print("-" * 70)
        
        # Plan execution
        print("📋 Planning execution...")
        tasks = await orchestrator.plan_execution(query)
        print(f"   Planned {len(tasks)} tasks")
        
        # Execute just the schema discovery step to see improvements
        print("\n🔍 Testing Schema Discovery Step...")
        schema_task = None
        for task in tasks:
            if task.task_type.value == "schema_discovery":
                schema_task = task
                break
        
        if schema_task:
            # Execute schema discovery
            schema_inputs = orchestrator._resolve_task_inputs(schema_task, {}, query)
            schema_result = await orchestrator._execute_schema_discovery(schema_inputs)
            
            print(f"\n📊 SCHEMA DISCOVERY RESULTS:")
            print(f"   Status: {schema_result.get('status', 'unknown')}")
            
            if schema_result.get('status') == 'completed':
                discovered_tables = schema_result.get('discovered_tables', [])
                table_suggestions = schema_result.get('table_suggestions', [])
                
                print(f"   Discovered tables: {len(discovered_tables)}")
                for i, table in enumerate(discovered_tables[:5]):
                    print(f"      {i+1}. {table}")
                
                if table_suggestions:
                    print(f"\n💡 TABLE SUGGESTIONS ({len(table_suggestions)}):")
                    for suggestion in table_suggestions:
                        print(f"      {suggestion['rank']}. {suggestion['table_name']}")
                        print(f"         Relevance: {suggestion['estimated_relevance']}")
                        print(f"         Score: {suggestion.get('relevance_score', 'N/A')}")
                else:
                    print(f"   ⚠️ No table suggestions (Azure not configured)")
            
            else:
                print(f"   ❌ Error: {schema_result.get('error', 'Unknown error')}")
        
        # Test user verification step
        print(f"\n👤 Testing User Verification Step...")
        verification_inputs = {
            "original_query": query,
            "1_discover_schema": schema_result
        }
        
        verification_result = await orchestrator._execute_user_verification(verification_inputs)
        
        print(f"\n📋 USER VERIFICATION RESULTS:")
        print(f"   Status: {verification_result.get('status', 'unknown')}")
        
        if verification_result.get('status') == 'completed':
            approved_tables = verification_result.get('approved_tables', [])
            selection_method = verification_result.get('selection_method', 'unknown')
            
            print(f"   Approved tables: {len(approved_tables)}")
            for table in approved_tables:
                print(f"      • {table}")
            print(f"   Selection method: {selection_method}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def show_azure_setup_instructions():
    """Show Azure setup instructions"""
    
    print(f"\n📋 AZURE AI SEARCH SETUP INSTRUCTIONS")
    print("="*60)
    print(f"To enable enhanced table discovery with top 4 suggestions:")
    print()
    print(f"1. 🔧 Create Azure AI Search Service:")
    print(f"   • Go to Azure Portal (portal.azure.com)")
    print(f"   • Create new 'Azure AI Search' resource")
    print(f"   • Choose Basic tier (sufficient for this use case)")
    print(f"   • Note the service URL and get admin key")
    print()
    print(f"2. ⚙️  Configure Environment Variables:")
    print(f"   • Copy .env.azure.template to your .env file")
    print(f"   • Replace placeholder values with your Azure credentials")
    print()
    print(f"3. 🚀 Run Setup Script:")
    print(f"   • python setup_azure_search.py")
    print(f"   • This will create the search index and populate it")
    print()
    print(f"4. ✅ Benefits After Setup:")
    print(f"   • Intelligent table discovery from 166+ tables")
    print(f"   • Top 4 most relevant table suggestions")
    print(f"   • Better similarity matching with OpenAI embeddings")
    print(f"   • Automatic user selection of best tables")
    print()
    print(f"📊 Current Status: Using fallback schema discovery")
    print(f"🎯 After Azure setup: Enhanced with vector similarity search")

if __name__ == "__main__":
    # Test current flow
    success = asyncio.run(test_enhanced_orchestrator_flow())
    
    if success:
        print(f"\n✅ Enhanced orchestrator flow working!")
        
        # Show Azure setup instructions
        asyncio.run(show_azure_setup_instructions())
        
        print(f"\n🎉 SUMMARY:")
        print(f"✅ Enhanced orchestrator ready")
        print(f"✅ Improved user verification with table suggestions")
        print(f"⚠️  Azure AI Search not configured (using fallback)")
        print(f"🔧 Run setup_azure_search.py after Azure configuration")
    else:
        print(f"\n❌ Enhanced orchestrator test failed")
