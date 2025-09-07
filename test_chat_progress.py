#!/usr/bin/env python3

"""
Test script to verify chat UI progress tracking with orchestrator
"""

import asyncio
import json
import websockets
import requests
import time

async def test_chat_progress():
    """Test real-time progress in chat UI"""
    print("🧪 Testing Chat UI Progress Tracking")
    
    # Connect to WebSocket
    websocket_uri = "ws://localhost:8000/ws/progress"
    
    try:
        # Test query execution with progress
        print("\n1. Submitting test query...")
        
        # Start WebSocket listener
        async with websockets.connect(websocket_uri) as websocket:
            print("✅ Connected to WebSocket")
            
            # Submit query via HTTP
            query_data = {
                "query": "Show me top 5 players by points this season",
                "user_id": "test_user"
            }
            
            print("📤 Sending query...")
            response = requests.post(
                "http://localhost:8000/api/agent/query", 
                json=query_data,
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"❌ Query failed: {response.status_code} - {response.text}")
                return
            
            print("✅ Query submitted, waiting for progress messages...")
            
            # Listen for progress messages
            message_count = 0
            start_time = time.time()
            
            try:
                while True:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(
                        websocket.recv(), 
                        timeout=30.0
                    )
                    
                    message_count += 1
                    elapsed = time.time() - start_time
                    
                    print(f"\n📨 Message {message_count} (t={elapsed:.1f}s):")
                    
                    try:
                        data = json.loads(message)
                        print(f"   Raw: {json.dumps(data, indent=2)}")
                        
                        # Analyze message structure
                        if "type" in data:
                            print(f"   📋 Type: {data['type']}")
                            if "data" in data:
                                print(f"   📊 Data Stage: {data['data'].get('stage', 'unknown')}")
                        else:
                            print(f"   📋 Direct Stage: {data.get('stage', 'unknown')}")
                            print(f"   📊 Progress: {data.get('progress', 'N/A')}%")
                            print(f"   🎯 Step: {data.get('stepName', data.get('currentStep', 'N/A'))}")
                        
                    except json.JSONDecodeError:
                        print(f"   ⚠️ Non-JSON message: {message}")
                    
                    # Stop after getting several messages or timeout
                    if message_count >= 10 or elapsed > 45:
                        print(f"\n🏁 Stopping after {message_count} messages")
                        break
                        
            except asyncio.TimeoutError:
                print(f"\n⏰ Timeout after {message_count} messages")
            except websockets.exceptions.ConnectionClosed:
                print(f"\n🔌 WebSocket connection closed after {message_count} messages")
            
            print(f"\n📊 Test Summary:")
            print(f"   - Received {message_count} progress messages")
            print(f"   - Total time: {time.time() - start_time:.1f}s")
            
            if message_count > 0:
                print("✅ Progress messages are being sent!")
            else:
                print("❌ No progress messages received")
                
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        print("💡 Make sure the backend server is running on localhost:8000")

if __name__ == "__main__":
    asyncio.run(test_chat_progress())
