#!/usr/bin/env python3
"""
Check Snowflake Connection Configuration
"""
import os
from dotenv import load_dotenv

def check_snowflake_config():
    load_dotenv()
    
    print('🔍 Checking Snowflake Environment Variables:')
    snowflake_vars = [
        'SNOWFLAKE_USER', 
        'SNOWFLAKE_PASSWORD', 
        'SNOWFLAKE_ACCOUNT', 
        'SNOWFLAKE_WAREHOUSE', 
        'SNOWFLAKE_DATABASE', 
        'SNOWFLAKE_SCHEMA'
    ]
    
    config_complete = True
    
    for var in snowflake_vars:
        value = os.getenv(var)
        if value:
            if 'PASSWORD' in var:
                print(f'✅ {var}: ***hidden***')
            else:
                display_value = value[:20] + '...' if len(value) > 20 else value
                print(f'✅ {var}: {display_value}')
        else:
            print(f'❌ {var}: Not set')
            config_complete = False
    
    print(f'\n🔧 DB_ENGINE: {os.getenv("DB_ENGINE", "not set")}')
    
    return config_complete

if __name__ == "__main__":
    complete = check_snowflake_config()
    
    if complete:
        print(f"\n✅ Snowflake configuration complete!")
        print(f"🔄 System should connect to live Snowflake database")
    else:
        print(f"\n❌ Snowflake configuration incomplete")
        print(f"💡 Please set missing environment variables in .env file")
