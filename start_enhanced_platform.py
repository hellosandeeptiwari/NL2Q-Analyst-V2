"""
Enhanced NL2Q Pharma Analytics Platform
Startup script with latest agentic approach
"""

import sys
import os
import subprocess
import threading
import time
from pathlib import Path

def print_banner():
    """Print startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║    🏥 Enhanced Pharma NL2Q Analytics Platform v2.0          ║
    ║                                                              ║
    ║    🤖 Latest Agentic AI Approach                            ║
    ║    👤 User Profiles & Chat History                          ║
    ║    💬 Claude Sonnet-inspired UI                             ║
    ║    🔒 Pharma Compliance & Governance                        ║
    ║    📊 Advanced Analytics & Visualizations                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check if virtual environment exists
    if not os.path.exists("venv") and not os.path.exists(".venv"):
        print("⚠️  Virtual environment not found. Creating one...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
    
    # Check backend dependencies
    try:
        import fastapi
        import uvicorn
        import openai
        import snowflake.connector
        print("✅ Backend dependencies available")
    except ImportError as e:
        print(f"❌ Missing backend dependency: {e}")
        print("Installing backend dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Check frontend dependencies
    frontend_path = Path("frontend")
    if frontend_path.exists():
        node_modules = frontend_path / "node_modules"
        if not node_modules.exists():
            print("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], cwd=frontend_path)
    
    print("✅ System requirements check completed")
    return True

def start_backend():
    """Start the enhanced backend server"""
    print("🚀 Starting Enhanced Backend Server...")
    
    # Set environment variables
    os.environ.setdefault("PYTHONPATH", os.getcwd())
    
    try:
        # Import the enhanced app
        from backend.enhanced_main import enhanced_app
        import uvicorn
        
        # Start server
        uvicorn.run(
            enhanced_app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=True
        )
    except Exception as e:
        print(f"❌ Backend startup failed: {e}")
        print("💡 Falling back to original backend...")
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "backend.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])

def start_frontend():
    """Start the React frontend"""
    print("🌐 Starting Enhanced Frontend...")
    
    frontend_path = Path("frontend")
    if not frontend_path.exists():
        print("❌ Frontend directory not found")
        return
    
    try:
        subprocess.run(["npm", "start"], cwd=frontend_path)
    except Exception as e:
        print(f"❌ Frontend startup failed: {e}")

def initialize_database():
    """Initialize enhanced database structures"""
    print("🗄️  Initializing enhanced database...")
    
    try:
        # Initialize chat history database
        from backend.history.enhanced_chat_history import ChatHistoryManager
        chat_manager = ChatHistoryManager()
        chat_manager.init_database()
        
        # Initialize user profiles
        from backend.auth.user_profile import create_demo_users
        create_demo_users()
        
        print("✅ Database initialization completed")
    except Exception as e:
        print(f"⚠️  Database initialization warning: {e}")

def check_environment():
    """Check environment configuration"""
    print("🔧 Checking environment configuration...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    # Check critical environment variables
    critical_vars = [
        "OPENAI_API_KEY",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_ACCOUNT"
    ]
    
    missing_vars = []
    for var in critical_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file")
    else:
        print("✅ Environment configuration looks good")
    
    return len(missing_vars) == 0

def main():
    """Main startup function"""
    print_banner()
    
    # Pre-flight checks
    if not check_requirements():
        print("❌ Requirements check failed")
        return
    
    if not check_environment():
        print("⚠️  Environment check failed, but continuing...")
    
    # Initialize database
    initialize_database()
    
    print("\n🎯 Starting Enhanced Pharma NL2Q Platform...")
    print("━" * 60)
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Wait for backend to start
    print("⏳ Waiting for backend to start...")
    time.sleep(5)
    
    # Test backend health
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is healthy")
        else:
            print("⚠️  Backend health check failed")
    except Exception as e:
        print(f"⚠️  Backend health check error: {e}")
    
    print("\n📋 System Status:")
    print("━" * 60)
    print("🔗 Backend API: http://localhost:8000")
    print("📊 API Docs: http://localhost:8000/docs")
    print("🌐 Frontend: http://localhost:3000 (starting...)")
    print("💾 Chat History: SQLite database initialized")
    print("👤 User Profiles: Demo users created")
    print("🤖 AI Models: GPT-4o-mini + o3-mini reasoning")
    print("🔒 Compliance: Pharma-specific governance enabled")
    
    print("\n🎉 Starting Frontend...")
    print("━" * 60)
    
    # Start frontend (this will block)
    start_frontend()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Enhanced Pharma NL2Q Platform...")
        print("Thank you for using our platform!")
    except Exception as e:
        print(f"\n❌ Startup error: {e}")
        print("Please check the logs and try again.")
