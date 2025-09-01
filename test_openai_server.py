#!/usr/bin/env python3
"""
Simple test server to verify OpenAI vector embeddings work
"""
import sys
from pathlib import Path
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import os

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from dotenv import load_dotenv
load_dotenv()

# Initialize database
from db.engine import get_adapter
adapter = get_adapter()

# Initialize Agent Orchestrator
from agents.orchestrator import AgentOrchestrator
agent_orchestrator = AgentOrchestrator()

def test_openai_integration():
    """Test OpenAI API integration"""
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"🔑 OpenAI API Key: {'✅ Found' if api_key else '❌ Not found'}")
    
    if api_key:
        print(f"🔑 API Key starts with: {api_key[:8]}...")
    
    # Test embedding
    try:
        from agents.openai_vector_matcher import OpenAIVectorMatcher
        matcher = OpenAIVectorMatcher()
        
        if matcher.api_key:
            print("✅ OpenAI Vector Matcher initialized with API key")
            
            # Test a simple embedding
            import openai
            response = openai.Embedding.create(
                input="test embedding",
                model="text-embedding-3-small"
            )
            print(f"✅ OpenAI API test successful! Embedding dimension: {len(response['data'][0]['embedding'])}")
        else:
            print("❌ No API key in vector matcher")
            
    except Exception as e:
        print(f"❌ OpenAI integration test failed: {e}")

def initialize_agents():
    """Initialize the agent system"""
    try:
        print("🔧 Initializing agents with database schema...")
        agent_orchestrator.initialize(adapter)
        print("✅ Agents initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False

class TestHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    
    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()
    
    def do_GET(self):
        if self.path == '/test-openai':
            # Test OpenAI integration
            try:
                test_openai_integration()
                
                status = agent_orchestrator.get_system_status()
                
                response = {
                    "openai_test": "completed",
                    "api_key_available": bool(os.getenv('OPENAI_API_KEY')),
                    "system_status": status
                }
                
                self._send_json_response(response)
            except Exception as e:
                self._send_json_response({"error": str(e)}, 500)
        
        elif self.path == '/test-search':
            # Test vector search
            try:
                result = agent_orchestrator.intelligent_table_suggestion(
                    "Find NBA basketball data", 3
                )
                self._send_json_response(result)
            except Exception as e:
                self._send_json_response({"error": str(e)}, 500)
        
        else:
            self._send_json_response({"error": "Not found"}, 404)
    
    def _send_json_response(self, data, status_code=200):
        self.send_response(status_code)
        self._send_cors_headers()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))

def run_test_server(port=8004):
    """Run test server"""
    print("🔧 Testing OpenAI integration...")
    test_openai_integration()
    
    print("\n🔧 Initializing agents...")
    if initialize_agents():
        print("✅ Ready for testing")
    else:
        print("⚠️ Running with limited functionality")
    
    server_address = ('', port)
    httpd = HTTPServer(server_address, TestHandler)
    
    print(f"\n🧪 Test Server running on http://localhost:{port}")
    print(f"🔗 Test endpoints:")
    print(f"   GET  http://localhost:{port}/test-openai")
    print(f"   GET  http://localhost:{port}/test-search")
    print("\n⏹️  Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Test server stopped")
        httpd.server_close()

if __name__ == "__main__":
    run_test_server()
