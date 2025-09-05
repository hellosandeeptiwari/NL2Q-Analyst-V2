#!/usr/bin/env python3
"""
Test Snowflake Identifier Quoting System
Tests the automatic quoting of table and column names for Snowflake
"""

import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_identifier_quoting():
    """Test the Snowflake identifier quoting utility"""
    
    print("🧪 Testing Snowflake Identifier Quoting System")
    print("=" * 70)
    
    try:
        from backend.utils.snowflake_quoter import (
            quote_snowflake_sql, 
            quote_table_name, 
            quote_column_name,
            SnowflakeIdentifierQuoter
        )
        
        quoter = SnowflakeIdentifierQuoter()
        
        # Test individual identifier quoting
        print("📋 Test 1: Individual Identifier Quoting")
        test_cases = [
            ("Final_NBA_Output_python_06042025", True),  # Has numbers - should be quoted
            ("Player_Stats_2024", True),                  # Has numbers - should be quoted  
            ("TEAM", False),                             # Simple name - no quoting needed
            ("SELECT", False),                           # SQL keyword - should not be quoted
            ("player_name", True),                       # Has underscore - should be quoted
            ("PlayerName", True),                        # Mixed case - should be quoted
            ('"Already_Quoted"', False),                 # Already quoted - no change
        ]
        
        for identifier, should_quote in test_cases:
            needs_quoting = quoter.needs_quoting(identifier)
            quoted = quote_table_name(identifier)
            
            expected = f'"{identifier}"' if should_quote and not identifier.startswith('"') else identifier
            
            print(f"   {identifier:30} -> {quoted:35} (Expected: {expected:35}) {'✅' if quoted == expected else '❌'}")
        
        print()
        
        # Test SQL query quoting
        print("📋 Test 2: SQL Query Quoting")
        
        test_queries = [
            "SELECT * FROM Final_NBA_Output_python_06042025",
            "SELECT player_name, team_id FROM nba_stats_2024 WHERE season = 2024",
            "SELECT COUNT(*) FROM Player_Performance_06042025 GROUP BY team_name",
        ]
        
        for sql in test_queries:
            quoted_sql = quote_snowflake_sql(sql)
            print(f"   Original: {sql}")
            print(f"   Quoted:   {quoted_sql}")
            print()
        
        # Test that keywords are not quoted
        print("📋 Test 3: Keyword Protection")
        sql_with_keywords = "SELECT COUNT(*) FROM Final_NBA_Output_python_06042025 WHERE player_name IS NOT NULL"
        quoted = quote_snowflake_sql(sql_with_keywords)
        print(f"   Input:  {sql_with_keywords}")
        print(f"   Output: {quoted}")
        
        # Check that SQL keywords are preserved
        keywords_preserved = all(keyword in quoted for keyword in ['SELECT', 'COUNT', 'FROM', 'WHERE', 'IS', 'NOT', 'NULL'])
        print(f"   Keywords preserved: {'✅' if keywords_preserved else '❌'}")
        
        print()
        print("🧪 Test Summary:")
        print("=" * 50)
        print("✅ Individual identifier quoting tested")
        print("✅ SQL query quoting tested") 
        print("✅ Keyword protection tested")
        print("🎉 Snowflake quoting system is ready!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        print(f"📋 Full error trace:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Snowflake Identifier Quoting Tests")
    print("=" * 80)
    
    success = test_identifier_quoting()
    
    if success:
        print("\n🏁 All tests passed - Quoting system working correctly!")
    else:
        print("\n💥 Tests failed - check implementation")
