#!/usr/bin/env python3
"""
Simple test script to start the NL2Q Agent backend and test the NBA table query
"""
import sys
import os
import requests
import time
import subprocess
import signal
import threading

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def start_server():
    """Start the FastAPI server in a separate process"""
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))

    cmd = [sys.executable, 'backend/main.py']
    process = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return process

def wait_for_server(timeout=30):
    """Wait for the server to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get('http://localhost:8000/health', timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def test_nba_query():
    """Test the NBA table query"""
    try:
        # Test the health endpoint
        response = requests.get('http://localhost:8000/health')
        print(f"✅ Health check: {response.status_code}")
        print(f"Response: {response.json()}")

        # Test the NBA query
        query_data = {
            "natural_language": "read table Final_NBA_Output_python_20250519 and create a visualization with frequency of recommended message and provider input",
            "job_id": "test_nba_001",
            "db_type": "snowflake"
        }

        headers = {"Authorization": "Bearer your_auth_token"}

        print("\n📊 Testing NBA table query...")
        response = requests.post('http://localhost:8000/query', json=query_data, headers=headers)

        if response.status_code == 200:
            result = response.json()
            print("✅ Query successful!")
            print(f"Generated SQL: {result.get('sql', 'N/A')}")
            print(f"Rows returned: {len(result.get('rows', []))}")
            if 'plotly_spec' in result and result['plotly_spec']:
                print("📈 Visualization spec generated!")
            else:
                print("⚠️  No visualization spec generated")
        else:
            print(f"❌ Query failed: {response.status_code}")
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"❌ Test failed: {e}")

def main():
    print("🚀 Starting NL2Q Agent Backend...")

    # Start the server
    server_process = start_server()

    try:
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        if wait_for_server():
            print("✅ Server is ready!")
            test_nba_query()
        else:
            print("❌ Server failed to start within timeout")

            # Print server output for debugging
            stdout, stderr = server_process.communicate()
            print("Server stdout:", stdout)
            print("Server stderr:", stderr)

    finally:
        # Clean up
        print("\n🛑 Stopping server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    main()
