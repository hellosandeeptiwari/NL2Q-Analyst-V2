#!/usr/bin/env python3

import asyncio
from backend.governance.rbac_manager import RBACManager

async def test_rbac_state():
    """Test RBAC manager state"""
    
    print("🔍 Testing RBAC Manager state...")
    
    # Create fresh RBAC manager 
    rbac = RBACManager()
    
    print(f"Available users: {list(rbac.users.keys())}")
    
    # Test each user
    for user_id in rbac.users.keys():
        user = rbac.users[user_id]
        print(f"User {user_id}: active={user.active}, roles={user.roles}")
        
        # Test permission check
        permission_check = await rbac.check_query_permissions(
            user_id=user_id,
            sql="SELECT * FROM \"Final_NBA_Output_python_06042025\" LIMIT 5"
        )
        print(f"  Permission check: {permission_check}")
    
    # Test the exact user_id being used
    print("\n🔍 Testing exact user_id 'default_user':")
    permission_check = await rbac.check_query_permissions(
        user_id="default_user",
        sql="SELECT * FROM \"Final_NBA_Output_python_06042025\" LIMIT 5"
    )
    print(f"Permission check for 'default_user': {permission_check}")

if __name__ == "__main__":
    asyncio.run(test_rbac_state())
