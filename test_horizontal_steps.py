import requests
import json
import time

def test_dynamic_horizontal_steps():
    """Test the new dynamic horizontal steps interface"""
    url = "http://localhost:8000/api/agent/query"
    
    # Test query that will go through all 7 steps
    test_query = "show me a visualization of top performers"
    
    payload = {
        "query": test_query,
        "db_config": {
            "host": "localhost",
            "port": 5432,
            "database": "nba_database",
            "username": "postgres",
            "password": "password"
        }
    }
    
    print("🎯 Testing Dynamic Horizontal Steps UI")
    print("=" * 50)
    print(f"Query: {test_query}")
    print("Expected behavior:")
    print("  • Steps appear dynamically as they complete")
    print("  • Only current/completed steps are clickable")
    print("  • Remaining steps counter shows future steps")
    print("  • Auto-selection of current step")
    print("  • Smooth animations and transitions")
    print("=" * 50)
    
    try:
        print("🚀 Sending request to backend...")
        response = requests.post(url, json=payload, timeout=120)
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Analyze execution plan
            if result.get("plan"):
                plan = result["plan"]
                print(f"\n✅ EXECUTION PLAN ANALYSIS:")
                print(f"   📋 Plan ID: {plan.get('plan_id', 'N/A')}")
                print(f"   🔄 Status: {plan.get('status', 'N/A')}")
                print(f"   📍 Current Step: {plan.get('current_step', 'N/A')}")
                print(f"   📊 Progress: {plan.get('progress', 'N/A')}%")
                
                if plan.get("reasoning_steps"):
                    print(f"   🧠 Reasoning Steps: {len(plan['reasoning_steps'])}")
                    for i, step in enumerate(plan['reasoning_steps'][:3], 1):
                        print(f"      {i}. {step}")
                    
                if plan.get("results"):
                    results = plan['results']
                    print(f"   📈 Available Results: {list(results.keys())}")
                    
                    # Step-by-step analysis
                    steps_completed = []
                    if 'schema_discovery' in results:
                        steps_completed.append("1. Schema Discovery ✅")
                    if 'semantic_understanding' in results:
                        steps_completed.append("2. Semantic Analysis ✅") 
                    if 'similarity_matching' in results:
                        steps_completed.append("3. Similarity Matching ✅")
                    if 'sql_query' in results:
                        steps_completed.append("4. Query Generation ✅")
                    if 'execution_result' in results:
                        steps_completed.append("5. Query Execution ✅")
                    if 'charts' in results:
                        steps_completed.append("6. Visualization ✅")
                        
                    print(f"\n   🎯 DYNAMIC STEPS PROGRESSION:")
                    for step in steps_completed:
                        print(f"      {step}")
            
            # Analyze charts
            if result.get("charts"):
                charts = result['charts']
                print(f"\n📈 CHARTS GENERATED: {len(charts)}")
                for i, chart in enumerate(charts, 1):
                    print(f"   Chart {i}: {chart.get('type', 'unknown')} - {chart.get('title', 'no title')}")
            
            # Analyze data
            if result.get("data"):
                print(f"\n📊 DATA RETURNED: {len(result['data'])} records")
                
            print(f"\n� FRONTEND UI EXPECTATIONS:")
            print(f"   🔄 Dynamic step reveal as each completes")
            print(f"   🎯 Auto-focus on current step")
            print(f"   📱 Smooth animations between steps") 
            print(f"   🔢 '+N more steps' indicator for remaining")
            print(f"   🖱️  Click interaction only on visible steps")
            print(f"   📋 Elegant plan display without technical terms")
            print(f"   ✨ Progressive disclosure of step details")
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_dynamic_horizontal_steps()
