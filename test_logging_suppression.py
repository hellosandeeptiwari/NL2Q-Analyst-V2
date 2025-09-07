#!/usr/bin/env python3
"""
Test WebSocket logging suppression
"""

import logging
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_logging_suppression():
    """Test that WebSocket logging is properly suppressed"""
    
    print("🧪 Testing WebSocket Logging Suppression")
    print("=" * 50)
    
    # Import and configure logging
    from backend.websocket_logging_config import configure_websocket_logging
    configure_websocket_logging()
    
    # Test various logger levels
    test_loggers = [
        "uvicorn.access",
        "uvicorn.protocols.websockets",
        "websockets.server",
        "websockets"
    ]
    
    print("\n📊 Logger Levels After Configuration:")
    for logger_name in test_loggers:
        logger = logging.getLogger(logger_name)
        level_name = logging.getLevelName(logger.level)
        print(f"  {logger_name}: {level_name}")
    
    # Test filter functionality
    print("\n🔍 Testing Message Filtering:")
    
    # Create test access logger
    access_logger = logging.getLogger("uvicorn.access")
    
    test_messages = [
        "INFO:     ('127.0.0.1', 64162) - \"WebSocket /ws/progress\" [accepted]",
        "INFO:     connection open",
        "INFO:     connection closed", 
        "INFO:     ('127.0.0.1', 8080) - \"GET /api/query\" - 200 OK",
        "INFO:     Regular HTTP request"
    ]
    
    for msg in test_messages:
        # Create a mock log record
        class MockRecord:
            def getMessage(self):
                return msg
        
        record = MockRecord()
        
        # Test each filter
        should_show = True
        for filter_obj in access_logger.filters:
            if not filter_obj.filter(record):
                should_show = False
                break
        
        status = "🔇 SUPPRESSED" if not should_show else "📢 SHOWN"
        print(f"  {status}: {msg}")
    
    print("\n✅ Logging suppression test completed!")
    print("\n💡 Expected behavior:")
    print("  - WebSocket connection messages should be SUPPRESSED")
    print("  - Regular API requests should be SHOWN")

if __name__ == "__main__":
    test_logging_suppression()
