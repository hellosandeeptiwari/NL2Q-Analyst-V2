#!/usr/bin/env python3
"""
Test the Enhanced Schema Intelligence System
"""

import asyncio
import sys
import os
sys.path.append('.')

from backend.agents.enhanced_schema_intelligence import EnhancedSchemaIntelligence

async def test_enhanced_schema():
    print('🧠 Testing Enhanced Schema Intelligence...')
    try:
        intelligence = EnhancedSchemaIntelligence()
        schema_data = await intelligence.discover_complete_schema()
        
        print(f'📊 Discovered {len(schema_data.get("tables", {}))} tables with deep analysis')
        print(f'🔗 Found {len(schema_data.get("relationships", []))} intelligent relationships')
        
        # Show table analysis
        print('\n📋 Table Analysis:')
        for table_name, table_data in schema_data.get('tables', {}).items():
            print(f'  • {table_name}:')
            print(f'    - Domain: {table_data.get("table_domain", "unknown")}')
            print(f'    - Purpose: {table_data.get("business_purpose", "unknown")}')
            print(f'    - Amount columns: {table_data.get("amount_columns", [])}')
            print(f'    - Key columns: {table_data.get("key_columns", [])}')
        
        # Show relationships
        print('\n🔗 Discovered Relationships:')
        for rel in schema_data.get('relationships', [])[:5]:  # Show first 5
            print(f'  • {rel.get("from_table")} -> {rel.get("to_table")} via {rel.get("from_column")}')
            print(f'    Type: {rel.get("relationship_type")}, Confidence: {rel.get("confidence", 0):.2f}')
            print(f'    Context: {rel.get("business_context", "N/A")}')
        
        # Show business intelligence
        if 'business_context' in schema_data:
            print('\n💡 Business Intelligence:')
            payment_guide = schema_data['business_context'].get('payment_analysis_guide', {})
            if 'primary_amount_sources' in payment_guide:
                print('   Primary payment amount sources:')
                for source in payment_guide['primary_amount_sources'][:3]:
                    print(f'     - {source["table"]}.{source["column"]}: {source["business_meaning"]}')
        
        print('\n✅ Enhanced Schema Intelligence test completed!')
        return schema_data
        
    except Exception as e:
        print(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(test_enhanced_schema())
