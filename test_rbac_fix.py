#!/usr/bin/env python3

import asyncio
from backend.governance.rbac_manager import RBACManager
from backend.tools.sql_runner import SQLRunner

async def test_rbac_and_sql():
    """Test RBAC and SQL execution with proper user"""
    
    print("🔍 Testing RBAC and SQL execution...")
    
    # Test RBAC manager
    rbac = RBACManager()
    
    # Check if default_user exists
    print(f"Available users: {list(rbac.users.keys())}")
    
    # Test permission check for default_user
    permission_check = await rbac.check_query_permissions(
        user_id="default_user",
        sql="SELECT * FROM \"Final_NBA_Output_python_06042025\" LIMIT 5"
    )
    
    print(f"Permission check for default_user: {permission_check}")
    
    # Test SQL execution
    if permission_check.get("allowed", False):
        sql_runner = SQLRunner()
        result = await sql_runner.execute_query(
            sql="SELECT * FROM \"Final_NBA_Output_python_06042025\" LIMIT 5",
            user_id="default_user"
        )
        
        print(f"SQL execution result: Success={result.success}")
        if result.success:
            print(f"Rows returned: {len(result.data)}")
            print(f"Columns: {result.columns}")
        else:
            print(f"Error: {result.error_message}")
    
    print("✅ Test completed")

if __name__ == "__main__":
    asyncio.run(test_rbac_and_sql())
