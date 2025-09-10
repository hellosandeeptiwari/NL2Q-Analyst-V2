"""
Debug Query History Saving
Test why the results key is not being saved properly
"""

import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.history.query_history import save_query_history, load_query_history

def test_save_history_with_results():
    print("🧪 Testing Query History Saving")
    print("=" * 50)
    
    # Test data
    test_results = [
        {"provider_name": "Provider A", "cost": 100, "state": "GA"},
        {"provider_name": "Provider B", "cost": 150, "state": "GA"},
        {"provider_name": "Provider C", "cost": 120, "state": "GA"}
    ]
    
    test_columns = ["provider_name", "cost", "state"]
    
    print(f"📊 Test data: {len(test_results)} rows")
    print(f"📋 Test columns: {test_columns}")
    print(f"🔍 Sample row: {test_results[0]}")
    
    # Save test query with results
    try:
        save_query_history(
            nl="Test query for debugging",
            sql="SELECT provider_name, cost, state FROM test_table",
            job_id="debug_test_001",
            user="debug_user",
            results=test_results,
            columns=test_columns
        )
        print("✅ save_query_history completed successfully")
        
        # Load and check what was saved
        history = load_query_history()
        print(f"📚 Total history entries: {len(history)}")
        
        # Find our test entry
        test_entry = None
        for entry in history:
            if entry.get('job_id') == 'debug_test_001':
                test_entry = entry
                break
        
        if test_entry:
            print(f"✅ Found test entry")
            print(f"🔍 Entry keys: {list(test_entry.keys())}")
            
            if 'results' in test_entry:
                print(f"✅ 'results' key exists")
                print(f"📊 Results data: {test_entry['results']}")
                print(f"📊 Results type: {type(test_entry['results'])}")
                print(f"📊 Results length: {len(test_entry['results']) if test_entry['results'] else 'None'}")
            else:
                print(f"❌ 'results' key missing!")
                print(f"🔍 Available keys: {list(test_entry.keys())}")
            
            if 'columns' in test_entry:
                print(f"✅ 'columns' key exists: {test_entry['columns']}")
            else:
                print(f"❌ 'columns' key missing!")
                
        else:
            print(f"❌ Test entry not found in history")
            print(f"🔍 Available job_ids: {[entry.get('job_id') for entry in history[-5:]]}")
        
    except Exception as e:
        print(f"❌ Error saving query history: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_save_history_with_results()
