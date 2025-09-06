import requests
import json

# Test the error feedback mechanism
def test_python_visualization():
    url = "http://localhost:8000/api/agent/query"
    
    # Query that should trigger Python visualization
    test_query = "show me a bar chart of player statistics"
    
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
    
    print(f"Testing query: {test_query}")
    print("Sending request to backend...")
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if we have charts
            if result.get("charts"):
                print(f"✅ Success! Generated {len(result['charts'])} charts")
                for i, chart in enumerate(result['charts']):
                    print(f"  Chart {i+1}: {chart.get('type', 'unknown')} - {chart.get('title', 'no title')}")
            else:
                print("❌ No charts generated")
                
            # Check for any errors in the response
            if result.get("error"):
                print(f"Error in response: {result['error']}")
                
            # Print execution details if available
            if result.get("execution_details"):
                details = result["execution_details"]
                print(f"Execution details: {details}")
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_python_visualization()
