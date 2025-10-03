#!/usr/bin/env python3
"""
Test our SQL generation fixes
"""

import sys
import os
import asyncio
sys.path.append('.')

from backend.nl2sql.enhanced_generator import generate_sql, GuardrailConfig

def test_sql_generation():
    """Test if our fixes generate proper SQL"""
    
    # Mock schema snapshot with TirosintTargetFlag
    schema_snapshot = {
        'tables': [
            {
                'name': 'Reporting_BI_PrescriberOverview',
                'columns': [
                    {'name': 'PrescriberName', 'data_type': 'varchar'},
                    {'name': 'TirosintTargetFlag', 'data_type': 'varchar'},
                    {'name': 'TRX(C4 Wk)', 'data_type': 'decimal'},
                    {'name': 'RegionName', 'data_type': 'varchar'},
                    {'name': 'TerritoryName', 'data_type': 'varchar'}
                ]
            }
        ]
    }
    
    # Create basic guardrail config
    constraints = GuardrailConfig(
        max_row_limit=1000,
        allowed_operations=['SELECT'],
        blocked_tables=[],
        require_where_clause=False,
        small_cell_threshold=5
    )
    
    test_query = "Show me patients with Tirosint target flag analysis"
    
    print(f"🧪 Testing SQL generation for: {test_query}")
    
    try:
        result = generate_sql(
            natural_language=test_query,
            schema_snapshot=schema_snapshot,
            constraints=constraints
        )
        
        print(f"✅ Generated SQL:")
        print(result.sql)
        print(f"\n📝 Rationale: {result.rationale}")
        
        # Check for our fixes
        if 'TirosintTargetFlag' in result.sql:
            print("✅ TirosintTargetFlag column found")
        else:
            print("❌ TirosintTargetFlag column missing")
            
        if "'Y'" in result.sql:
            print("✅ Using 'Y' for flag (correct)")
        elif "= 1" in result.sql:
            print("❌ Still using = 1 for flag (incorrect)")
        else:
            print("⚠️ No flag condition found")
            
        if '[TirosintTargetFlag]' in result.sql:
            print("✅ Column bracketing working")
        else:
            print("⚠️ Column bracketing not found")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sql_generation()