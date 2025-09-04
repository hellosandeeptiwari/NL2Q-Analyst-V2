"""
Enhanced NL2Q Platform Test Suite
Tests the enhanced features including user profiles, chat history, and agentic orchestrator
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

async def test_user_profiles():
    """Test user profile management"""
    print("🧪 Testing User Profile Management...")
    
    try:
        from backend.auth.user_profile import profile_manager, UserRole
        
        # Test creating a user
        profile = profile_manager.create_user(
            username="test_user",
            email="test@pharma.com",
            full_name="Test User",
            role=UserRole.ANALYST,
            department="Test Department",
            therapeutic_areas=["Oncology", "Diabetes"]
        )
        
        # Test retrieving user
        retrieved = profile_manager.get_user(profile.user_id)
        assert retrieved is not None
        assert retrieved.username == "test_user"
        
        # Test updating preferences
        profile_manager.update_user_preferences(profile.user_id, {
            "theme": "dark",
            "auto_execute_high_confidence": False
        })
        
        print("✅ User Profile Management: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ User Profile Management: FAILED - {e}")
        return False

async def test_chat_history():
    """Test chat history management"""
    print("🧪 Testing Chat History Management...")
    
    try:
        from backend.history.enhanced_chat_history import (
            ChatHistoryManager, MessageType, MessageStatus
        )
        
        chat_manager = ChatHistoryManager(db_path="./test_chat.db")
        
        # Test creating conversation
        conversation = chat_manager.create_conversation(
            user_id="test_user_123",
            title="Test Conversation",
            therapeutic_area="Oncology"
        )
        
        # Test adding messages
        message = chat_manager.add_message(
            conversation_id=conversation.conversation_id,
            user_id="test_user_123",
            message_type=MessageType.USER_QUERY,
            content="Show oncology drug sales",
            metadata={"test": True}
        )
        
        # Test retrieving conversation
        retrieved_conv = chat_manager.get_conversation(conversation.conversation_id)
        assert retrieved_conv is not None
        assert len(retrieved_conv.messages) == 1
        
        # Test search
        search_results = chat_manager.search_conversations("test_user_123", "oncology")
        assert len(search_results) >= 1
        
        # Clean up test database - close any connections first
        del chat_manager  # This should close the connection
        import time
        time.sleep(0.1)  # Brief delay to ensure file is released
        
        if os.path.exists("./test_chat.db"):
            try:
                os.unlink("./test_chat.db")
            except PermissionError:
                # If file is still locked, just move on
                pass
        
        print("✅ Chat History Management: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Chat History Management: FAILED - {e}")
        return False

async def test_enhanced_orchestrator():
    """Test enhanced agentic orchestrator"""
    print("🧪 Testing Enhanced Agentic Orchestrator...")
    
    try:
        # Set a mock API key for testing
        os.environ["OPENAI_API_KEY"] = "test-key-for-testing"
        
        from backend.agents.enhanced_orchestrator import (
            EnhancedAgenticOrchestrator, PharmaQueryContext, 
            PharmaComplianceLevel, TherapeuticContext
        )
        
        orchestrator = EnhancedAgenticOrchestrator()
        
        # Test building pharma context
        test_context = PharmaQueryContext(
            therapeutic_areas=["Oncology"],
            compliance_level=PharmaComplianceLevel.INTERNAL,
            user_role="analyst",
            department="Commercial Analytics",
            data_permissions=["ENHANCED_NBA.*"],
            conversation_history=[],
            regulatory_flags=[],
            business_context={}
        )
        
        # Test cache key generation
        cache_key = orchestrator._generate_pharma_cache_key(
            "test query", "user_123", test_context
        )
        assert cache_key is not None
        assert len(cache_key) == 32  # MD5 hash length
        
        print("✅ Enhanced Agentic Orchestrator: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Agentic Orchestrator: FAILED - {e}")
        return False

async def test_api_endpoints():
    """Test enhanced API endpoints"""
    print("🧪 Testing Enhanced API Endpoints...")
    
    try:
        # Set a mock API key for testing
        os.environ["OPENAI_API_KEY"] = "test-key-for-testing"
        
        from backend.enhanced_main import create_enhanced_app
        from fastapi.testclient import TestClient
        
        app = create_enhanced_app()
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        
        # Test therapeutic areas endpoint
        response = client.get("/api/pharma/therapeutic-areas")
        assert response.status_code == 200
        data = response.json()
        assert "therapeutic_areas" in data
        
        print("✅ Enhanced API Endpoints: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced API Endpoints: FAILED - {e}")
        return False

async def test_frontend_components():
    """Test frontend component structure"""
    print("🧪 Testing Frontend Components...")
    
    try:
        # Check if enhanced components exist
        frontend_components = [
            "frontend/src/components/EnhancedPharmaChat.tsx",
            "frontend/src/components/EnhancedPharmaChat.css",
        ]
        
        for component in frontend_components:
            if not os.path.exists(component):
                raise FileNotFoundError(f"Component not found: {component}")
        
        # Check package.json for required dependencies
        with open("frontend/package.json", "r") as f:
            package_data = json.load(f)
        
        required_deps = ["react-icons", "react-plotly.js", "axios"]
        for dep in required_deps:
            if dep not in package_data.get("dependencies", {}):
                raise ValueError(f"Missing dependency: {dep}")
        
        print("✅ Frontend Components: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Frontend Components: FAILED - {e}")
        return False

async def run_all_tests():
    """Run all test suites"""
    print("🚀 Starting Enhanced NL2Q Platform Test Suite")
    print("=" * 60)
    
    test_functions = [
        test_user_profiles,
        test_chat_history,
        test_enhanced_orchestrator,
        test_api_endpoints,
        test_frontend_components
    ]
    
    results = []
    for test_func in test_functions:
        result = await test_func()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    print("🏁 Test Suite Summary")
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced platform is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

def main():
    """Main test runner"""
    try:
        # Set up test environment
        os.environ.setdefault("PYTHONPATH", os.getcwd())
        
        # Run tests
        success = asyncio.run(run_all_tests())
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
