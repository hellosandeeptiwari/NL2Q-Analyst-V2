#!/usr/bin/env python3
"""
Force complete reindex with proper Azure SQL schema extraction
"""

import sys
import os
from pathlib import Path
import requests
import json

def main():
    print("🔄 Forcing complete reindex with correct Azure SQL schema")
    print("=" * 80)
    
    # Backend server URL
    base_url = "http://localhost:8000"
    
    try:
        # 1. First clear the entire Pinecone index
        print("\n🗑️ Step 1: Clearing Pinecone index...")
        clear_url = f"{base_url}/database/clear-pinecone"
        response = requests.post(clear_url)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Index cleared: {result}")
        else:
            print(f"❌ Failed to clear index: {response.status_code} - {response.text}")
            return
        
        # 2. Force reindex all tables with proper schema extraction
        print("\n📊 Step 2: Force reindexing all tables...")
        reindex_url = f"{base_url}/database/reindex-schema"
        
        # Payload for complete reindex
        payload = {
            "force_clear": True,
            "selected_tables": ["Reporting_BI_NGD", "Reporting_BI_PrescriberOverview", "Reporting_BI_PrescriberProfile"]
        }
        
        response = requests.post(reindex_url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Reindexing successful:")
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Message: {result.get('message', 'no message')}")
            
            if 'indexing_results' in result:
                for table, info in result['indexing_results'].items():
                    print(f"   📊 {table}: {info.get('vectors_created', 0)} vectors created")
        else:
            print(f"❌ Reindexing failed: {response.status_code} - {response.text}")
            return
        
        # 3. Verify the reindexing worked
        print("\n🔍 Step 3: Verifying reindexing...")
        status_url = f"{base_url}/database/pinecone-status"
        response = requests.get(status_url)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Verification:")
            print(f"   Total vectors: {result.get('total_vectors', 0)}")
            print(f"   Index dimension: {result.get('dimension', 0)}")
            
            if 'sample_metadata' in result:
                print(f"   Sample table: {result['sample_metadata'].get('table_name', 'unknown')}")
                columns = result['sample_metadata'].get('columns', [])
                print(f"   Sample columns ({len(columns)}): {columns[:5]}...")
        else:
            print(f"❌ Verification failed: {response.status_code} - {response.text}")
        
        print("\n✅ Complete reindex finished!")
        print("🎯 Now test the retry logic again with correct schema")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()