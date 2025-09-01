#!/usr/bin/env python3
"""
Debug agent initialization issue
"""
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from dotenv import load_dotenv
load_dotenv()

def debug_agent_init():
    """Debug the agent initialization issue"""
    print("🔍 Debugging agent initialization...")
    
    try:
        # Step 1: Get database adapter
        print("1. Getting database adapter...")
        from db.engine import get_adapter
        adapter = get_adapter()
        print(f"✅ Adapter: {type(adapter)}")
        print(f"   Has 'run' method: {hasattr(adapter, 'run')}")
        
        # Step 2: Test adapter directly
        print("2. Testing adapter run method...")
        result = adapter.run("SELECT 1 as test")
        print(f"✅ Direct adapter test: {result.rows if result else 'No result'}")
        
        # Step 3: Initialize orchestrator
        print("3. Creating orchestrator...")
        from agents.orchestrator import AgentOrchestrator
        orchestrator = AgentOrchestrator(openai_api_key=os.getenv('OPENAI_API_KEY'))
        print(f"✅ Orchestrator created")
        print(f"   Vector matcher: {type(orchestrator.vector_matcher) if orchestrator.vector_matcher else 'None'}")
        
        # Step 4: Test initialization with our adapter
        print("4. Testing orchestrator initialization...")
        if orchestrator.vector_matcher:
            print("   Calling initialize_from_database...")
            
            # Let's check what the vector matcher receives
            print(f"   Adapter type being passed: {type(adapter)}")
            
            # Call the vector matcher method directly
            try:
                orchestrator.vector_matcher.initialize_from_database(adapter, force_rebuild=False)
                print("✅ Vector matcher initialized successfully")
            except Exception as e:
                print(f"❌ Vector matcher failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ No vector matcher available")
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_agent_init()
