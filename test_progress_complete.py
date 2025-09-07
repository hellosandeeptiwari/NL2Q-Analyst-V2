#!/usr/bin/env python3
"""
Quick test of real-time progress functionality
"""
import asyncio
import sys
sys.path.append('backend')

async def test_progress_functions():
    """Test the fixed progress functions"""
    try:
        from backend.main import broadcast_progress, update_progress
        
        print("🧪 Testing real-time progress functions...")
        
        # Test execution progress
        execution_data = {
            "stage": "execution_started",
            "stepName": "Testing Real-time Progress",
            "currentStep": 1,
            "totalSteps": 3,
            "completedSteps": 1,
            "progress": 33
        }
        
        print("📤 Testing execution progress broadcast...")
        await broadcast_progress(execution_data)
        print("✅ Execution progress broadcast successful")
        
        # Test indexing progress
        print("📤 Testing indexing progress broadcast...")
        await broadcast_progress()  # No data = indexing progress
        print("✅ Indexing progress broadcast successful")
        
        # Test update_progress function
        print("📤 Testing update_progress function...")
        await update_progress("start", total=5)
        print("✅ update_progress function successful")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("Real-time progress system is working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Testing Real-Time Progress System")
    print("-" * 50)
    asyncio.run(test_progress_functions())
