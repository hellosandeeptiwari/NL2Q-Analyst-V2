#!/usr/bin/env python3
"""
Use real backend API to trigger complete reindexing
"""

import requests
import json

def main():
    print("🔄 Using real backend API for complete reindexing")
    print("=" * 80)
    
    # Backend server URL
    base_url = "http://localhost:8000"
    
    try:
        # 1. First check indexing status
        print("\n🔍 Step 1: Checking current indexing status...")
        status_url = f"{base_url}/api/database/indexing-status"
        response = requests.get(status_url)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Current status:")
            print(f"   Total vectors: {result.get('total_vectors', 0)}")
            print(f"   Indexed tables: {result.get('indexed_tables', [])}")
        else:
            print(f"❌ Failed to get status: {response.status_code} - {response.text}")
        
        # 2. Start complete reindexing with force_clear
        print("\n📊 Step 2: Starting complete reindexing...")
        reindex_url = f"{base_url}/api/database/start-indexing"
        
        # Payload for complete reindex - check what the API expects
        payload = {
            "force_clear": True,
            "selected_tables": ["Reporting_BI_NGD", "Reporting_BI_PrescriberOverview", "Reporting_BI_PrescriberProfile"]
        }
        
        print(f"   📤 Sending payload: {payload}")
        response = requests.post(reindex_url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Reindexing started:")
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Message: {result.get('message', 'no message')}")
            
            # Print detailed results if available
            if 'results' in result:
                for table_name, table_result in result['results'].items():
                    print(f"   📊 {table_name}:")
                    print(f"      Vectors: {table_result.get('vectors_created', 0)}")
                    print(f"      Status: {table_result.get('status', 'unknown')}")
        else:
            print(f"❌ Reindexing failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return
        
        # 3. Verify the reindexing worked
        print("\n🔍 Step 3: Verifying reindexing results...")
        response = requests.get(status_url)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ After reindexing:")
            print(f"   Total vectors: {result.get('total_vectors', 0)}")
            print(f"   Indexed tables: {result.get('indexed_tables', [])}")
            
            # Check if we have sample metadata
            if 'sample_metadata' in result:
                sample = result['sample_metadata']
                print(f"   Sample table: {sample.get('table_name', 'unknown')}")
                columns = sample.get('columns', [])
                print(f"   Sample columns ({len(columns)}): {columns[:5]}...")
        else:
            print(f"❌ Verification failed: {response.status_code} - {response.text}")
        
        print("\n✅ Complete reindex finished!")
        print("🎯 Now the retry logic should work with correct schema")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()