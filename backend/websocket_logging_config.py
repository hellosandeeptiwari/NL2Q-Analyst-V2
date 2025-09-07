"""
WebSocket Logging Suppression Configuration
Add this to your backend startup to reduce WebSocket connection noise
"""

import logging

def configure_websocket_logging():
    """Configure logging to suppress noisy WebSocket connection messages"""
    
    # Set log levels to reduce noise
    loggers_to_suppress = [
        "uvicorn.access",
        "uvicorn.protocols.websockets", 
        "uvicorn.protocols.websockets.websockets_impl",
        "websockets.server",
        "websockets.protocol",
        "websockets"
    ]
    
    for logger_name in loggers_to_suppress:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False
    
    # Create custom filter for remaining access logs
    class WebSocketFilter(logging.Filter):
        def filter(self, record):
            if hasattr(record, 'getMessage'):
                message = record.getMessage()
                # Suppress specific WebSocket messages
                suppress_patterns = [
                    "WebSocket /ws/progress",
                    "connection open",
                    "connection closed", 
                    "accepted",
                    "127.0.0.1",
                    "/ws/"
                ]
                for pattern in suppress_patterns:
                    if pattern in message:
                        return False
            return True
    
    # Apply filter to uvicorn access logger
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.addFilter(WebSocketFilter())
    
    print("🔇 WebSocket logging suppression configured")

if __name__ == "__main__":
    configure_websocket_logging()
    print("✅ Logging configuration applied")
