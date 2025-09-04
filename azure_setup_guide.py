#!/usr/bin/env python3
"""
Step-by-Step Azure AI Search Setup Guide
Follow these steps to enable enhanced schema discovery
"""

print("""
🚀 AZURE AI SEARCH SETUP GUIDE FOR NL2Q
========================================

This will enable intelligent table discovery with top 4 suggestions from your 166+ Snowflake tables.

📋 STEP-BY-STEP INSTRUCTIONS:

1. 🔧 CREATE AZURE AI SEARCH SERVICE
   ────────────────────────────────────
   • Go to: https://portal.azure.com
   • Click "Create a resource"
   • Search for "Azure AI Search"
   • Click "Create"
   
   Service Configuration:
   • Service name: nl2q-search-service (or your choice)
   • Subscription: Your Azure subscription
   • Resource group: Create new or existing
   • Location: Choose your region
   • Pricing tier: BASIC ($250/month - sufficient for this project)
   
   • Click "Review + Create"
   • Wait for deployment (2-3 minutes)

2. 📝 GET SERVICE CREDENTIALS
   ──────────────────────────
   • Go to your new Search service in Azure Portal
   • Copy the URL (e.g., https://nl2q-search-service.search.windows.net)
   • Click "Keys" in left menu
   • Copy the "Primary admin key"

3. ⚙️  UPDATE .ENV FILE
   ──────────────────────
   Replace these placeholders in your .env file:
   
   AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
   AZURE_SEARCH_KEY=your_admin_key_here
   AZURE_SEARCH_INDEX_NAME=nl2q-schema-index
   
   With your actual values:
   
   AZURE_SEARCH_ENDPOINT=https://nl2q-search-service.search.windows.net
   AZURE_SEARCH_KEY=your_actual_admin_key
   AZURE_SEARCH_INDEX_NAME=nl2q-schema-index

4. 📦 INSTALL REQUIRED PACKAGES
   ─────────────────────────────
   Run: pip install azure-search-documents==11.4.0 azure-core==1.29.5

5. 🚀 RUN SETUP SCRIPT
   ───────────────────
   Run: python setup_azure_search.py
   
   This will:
   • Create the search index
   • Connect to your Snowflake database
   • Analyze all 166+ tables
   • Create semantic chunks for each table
   • Generate OpenAI embeddings
   • Upload everything to Azure AI Search

6. ✅ TEST ENHANCED DISCOVERY
   ──────────────────────────
   Run: python test_enhanced_orchestrator.py
   
   You should see:
   • "Using Azure-Enhanced Schema Discovery"
   • Top 4 table suggestions with relevance scores
   • Automatic selection of best matching tables

🎯 BENEFITS AFTER SETUP:
========================
• Intelligent discovery from 166+ Snowflake tables
• Vector similarity search with OpenAI embeddings
• Top 4 most relevant table suggestions for any query
• Automatic ranking by relevance score
• Better handling of table name variations
• Semantic understanding of business context

💰 COST ESTIMATION:
==================
• Azure AI Search Basic: ~$250/month
• OpenAI embeddings: ~$0.02 per 1K tokens (one-time indexing cost ~$5-10)
• OpenAI search queries: ~$0.0001 per query

🔧 TROUBLESHOOTING:
==================
If setup fails:
• Check Azure credentials in .env file
• Verify Snowflake connection is working
• Ensure OpenAI API key has sufficient credits
• Check Python package versions

📞 SUPPORT:
===========
If you need help with Azure setup, the Azure AI Search documentation is at:
https://docs.microsoft.com/en-us/azure/search/

Ready to proceed? Follow the steps above, then run setup_azure_search.py!
""")

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("🔍 CURRENT CONFIGURATION CHECK:")
    print("="*40)
    
    # Check current configuration
    azure_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "NOT_SET")
    azure_key = os.getenv("AZURE_SEARCH_KEY", "NOT_SET")
    azure_index = os.getenv("AZURE_SEARCH_INDEX_NAME", "NOT_SET")
    openai_key = os.getenv("OPENAI_API_KEY", "NOT_SET")
    
    print(f"Azure Endpoint: {'✅ SET' if azure_endpoint != 'NOT_SET' and 'your-search' not in azure_endpoint else '❌ NEEDS SETUP'}")
    print(f"Azure Key: {'✅ SET' if azure_key != 'NOT_SET' and 'your_admin' not in azure_key else '❌ NEEDS SETUP'}")
    print(f"Azure Index: {'✅ SET' if azure_index != 'NOT_SET' else '❌ NEEDS SETUP'}")
    print(f"OpenAI Key: {'✅ SET' if openai_key != 'NOT_SET' else '❌ NEEDS SETUP'}")
    
    if all(var != 'NOT_SET' for var in [azure_endpoint, azure_key, azure_index, openai_key]) and 'your-search' not in azure_endpoint:
        print(f"\n🎉 Ready to run: python setup_azure_search.py")
    else:
        print(f"\n⚠️  Please complete Azure setup first")
