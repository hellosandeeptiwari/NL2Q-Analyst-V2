"""
Enhanced Chat History Management for Pharma NL2Q System
Supports conversation context, query templates, and pharma-specific insights
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import uuid
import os
from enum import Enum
import sqlite3

class MessageType(Enum):
    USER_QUERY = "user_query"
    SYSTEM_RESPONSE = "system_response"
    SQL_EXECUTION = "sql_execution"
    VISUALIZATION = "visualization"
    ERROR = "error"
    SUGGESTION = "suggestion"
    PLAN_UPDATE = "plan_update"

class MessageStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ChatMessage:
    """Individual chat message with rich metadata"""
    message_id: str
    conversation_id: str
    user_id: str
    message_type: MessageType
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    status: MessageStatus = MessageStatus.COMPLETED
    response_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    parent_message_id: Optional[str] = None  # For threading
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        msg_dict = asdict(self)
        msg_dict['message_type'] = self.message_type.value
        msg_dict['status'] = self.status.value
        msg_dict['timestamp'] = self.timestamp.isoformat()
        return msg_dict

@dataclass
class Conversation:
    """Complete conversation with context and analytics"""
    conversation_id: str
    user_id: str
    title: str
    created_at: datetime
    last_activity: datetime
    messages: List[ChatMessage]
    context: Dict[str, Any]
    tags: List[str]
    therapeutic_area: Optional[str] = None
    total_cost: float = 0.0
    total_tokens: int = 0
    is_favorite: bool = False
    is_archived: bool = False
    
    def add_message(self, message: ChatMessage):
        """Add message to conversation"""
        self.messages.append(message)
        self.last_activity = datetime.now()
        
        # Update aggregated metrics
        if message.cost_usd:
            self.total_cost += message.cost_usd
        if message.tokens_used:
            self.total_tokens += message.tokens_used
    
    def get_recent_context(self, max_messages: int = 5) -> List[ChatMessage]:
        """Get recent messages for context"""
        return self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        conv_dict = asdict(self)
        conv_dict['created_at'] = self.created_at.isoformat()
        conv_dict['last_activity'] = self.last_activity.isoformat()
        conv_dict['messages'] = [msg.to_dict() for msg in self.messages]
        return conv_dict

class ChatHistoryManager:
    """Enhanced chat history management with SQLite backend"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = "backend/history/chat_history.db"
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for chat history"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    context TEXT,
                    tags TEXT,
                    therapeutic_area TEXT,
                    total_cost REAL DEFAULT 0.0,
                    total_tokens INTEGER DEFAULT 0,
                    is_favorite INTEGER DEFAULT 0,
                    is_archived INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    timestamp TEXT NOT NULL,
                    status TEXT DEFAULT 'completed',
                    response_time_ms INTEGER,
                    tokens_used INTEGER,
                    cost_usd REAL,
                    parent_message_id TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_last_activity ON conversations(last_activity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
    
    def create_conversation(self, user_id: str, title: str = None, 
                          therapeutic_area: str = None) -> Conversation:
        """Create new conversation"""
        conversation_id = str(uuid.uuid4())
        now = datetime.now()
        
        if not title:
            title = f"Conversation {now.strftime('%Y-%m-%d %H:%M')}"
        
        conversation = Conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            title=title,
            created_at=now,
            last_activity=now,
            messages=[],
            context={},
            tags=[],
            therapeutic_area=therapeutic_area
        )
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO conversations 
                (conversation_id, user_id, title, created_at, last_activity, 
                 context, tags, therapeutic_area, total_cost, total_tokens, 
                 is_favorite, is_archived)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                conversation_id, user_id, title, now.isoformat(), now.isoformat(),
                json.dumps({}), json.dumps([]), therapeutic_area, 0.0, 0, 0, 0
            ))
        
        return conversation
    
    def add_message(self, conversation_id: str, user_id: str, 
                   message_type: MessageType, content: str,
                   metadata: Dict[str, Any] = None, 
                   status: MessageStatus = MessageStatus.COMPLETED,
                   response_time_ms: int = None, tokens_used: int = None,
                   cost_usd: float = None, parent_message_id: str = None) -> ChatMessage:
        """Add message to conversation"""
        message_id = str(uuid.uuid4())
        now = datetime.now()
        
        message = ChatMessage(
            message_id=message_id,
            conversation_id=conversation_id,
            user_id=user_id,
            message_type=message_type,
            content=content,
            metadata=metadata or {},
            timestamp=now,
            status=status,
            response_time_ms=response_time_ms,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            parent_message_id=parent_message_id
        )
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO messages 
                (message_id, conversation_id, user_id, message_type, content,
                 metadata, timestamp, status, response_time_ms, tokens_used,
                 cost_usd, parent_message_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message_id, conversation_id, user_id, message_type.value, content,
                json.dumps(metadata or {}), now.isoformat(), status.value,
                response_time_ms, tokens_used, cost_usd, parent_message_id
            ))
            
            # Update conversation last_activity and totals
            conn.execute("""
                UPDATE conversations 
                SET last_activity = ?,
                    total_cost = total_cost + COALESCE(?, 0),
                    total_tokens = total_tokens + COALESCE(?, 0)
                WHERE conversation_id = ?
            """, (now.isoformat(), cost_usd or 0, tokens_used or 0, conversation_id))
        
        return message
    
    def get_conversation(self, conversation_id: str, include_messages: bool = True) -> Optional[Conversation]:
        """Get conversation by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get conversation
            conv_row = conn.execute("""
                SELECT * FROM conversations WHERE conversation_id = ?
            """, (conversation_id,)).fetchone()
            
            if not conv_row:
                return None
            
            conversation = Conversation(
                conversation_id=conv_row['conversation_id'],
                user_id=conv_row['user_id'],
                title=conv_row['title'],
                created_at=datetime.fromisoformat(conv_row['created_at']),
                last_activity=datetime.fromisoformat(conv_row['last_activity']),
                messages=[],
                context=json.loads(conv_row['context'] or '{}'),
                tags=json.loads(conv_row['tags'] or '[]'),
                therapeutic_area=conv_row['therapeutic_area'],
                total_cost=conv_row['total_cost'],
                total_tokens=conv_row['total_tokens'],
                is_favorite=bool(conv_row['is_favorite']),
                is_archived=bool(conv_row['is_archived'])
            )
            
            if include_messages:
                # Get messages
                message_rows = conn.execute("""
                    SELECT * FROM messages 
                    WHERE conversation_id = ? 
                    ORDER BY timestamp ASC
                """, (conversation_id,)).fetchall()
                
                for msg_row in message_rows:
                    message = ChatMessage(
                        message_id=msg_row['message_id'],
                        conversation_id=msg_row['conversation_id'],
                        user_id=msg_row['user_id'],
                        message_type=MessageType(msg_row['message_type']),
                        content=msg_row['content'],
                        metadata=json.loads(msg_row['metadata'] or '{}'),
                        timestamp=datetime.fromisoformat(msg_row['timestamp']),
                        status=MessageStatus(msg_row['status']),
                        response_time_ms=msg_row['response_time_ms'],
                        tokens_used=msg_row['tokens_used'],
                        cost_usd=msg_row['cost_usd'],
                        parent_message_id=msg_row['parent_message_id']
                    )
                    conversation.messages.append(message)
        
        return conversation
    
    def get_user_conversations(self, user_id: str, limit: int = 50, 
                             include_archived: bool = False) -> List[Conversation]:
        """Get user's conversations"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = """
                SELECT * FROM conversations 
                WHERE user_id = ?
            """
            params = [user_id]
            
            if not include_archived:
                query += " AND is_archived = 0"
            
            query += " ORDER BY last_activity DESC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(query, params).fetchall()
            
            conversations = []
            for row in rows:
                conv = Conversation(
                    conversation_id=row['conversation_id'],
                    user_id=row['user_id'],
                    title=row['title'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    last_activity=datetime.fromisoformat(row['last_activity']),
                    messages=[],  # Don't load messages for list view
                    context=json.loads(row['context'] or '{}'),
                    tags=json.loads(row['tags'] or '[]'),
                    therapeutic_area=row['therapeutic_area'],
                    total_cost=row['total_cost'],
                    total_tokens=row['total_tokens'],
                    is_favorite=bool(row['is_favorite']),
                    is_archived=bool(row['is_archived'])
                )
                conversations.append(conv)
        
        return conversations
    
    def search_conversations(self, user_id: str, query: str, 
                           limit: int = 20) -> List[Conversation]:
        """Search conversations by content"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Search in conversation titles and message content
            rows = conn.execute("""
                SELECT DISTINCT c.* FROM conversations c
                LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                WHERE c.user_id = ? AND (
                    c.title LIKE ? OR 
                    m.content LIKE ?
                )
                ORDER BY c.last_activity DESC 
                LIMIT ?
            """, (user_id, f"%{query}%", f"%{query}%", limit)).fetchall()
            
            conversations = []
            for row in rows:
                conv = Conversation(
                    conversation_id=row['conversation_id'],
                    user_id=row['user_id'],
                    title=row['title'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    last_activity=datetime.fromisoformat(row['last_activity']),
                    messages=[],
                    context=json.loads(row['context'] or '{}'),
                    tags=json.loads(row['tags'] or '[]'),
                    therapeutic_area=row['therapeutic_area'],
                    total_cost=row['total_cost'],
                    total_tokens=row['total_tokens'],
                    is_favorite=bool(row['is_favorite']),
                    is_archived=bool(row['is_archived'])
                )
                conversations.append(conv)
        
        return conversations
    
    def update_conversation_title(self, conversation_id: str, title: str):
        """Update conversation title"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE conversations SET title = ? WHERE conversation_id = ?
            """, (title, conversation_id))
    
    def toggle_favorite(self, conversation_id: str):
        """Toggle conversation favorite status"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE conversations 
                SET is_favorite = 1 - is_favorite 
                WHERE conversation_id = ?
            """, (conversation_id,))
    
    def archive_conversation(self, conversation_id: str):
        """Archive conversation"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE conversations 
                SET is_archived = 1 
                WHERE conversation_id = ?
            """, (conversation_id,))
    
    def get_usage_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get user's usage analytics"""
        since_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Total conversations and messages
            total_convs = conn.execute("""
                SELECT COUNT(*) as count FROM conversations 
                WHERE user_id = ? AND created_at >= ?
            """, (user_id, since_date.isoformat())).fetchone()['count']
            
            total_msgs = conn.execute("""
                SELECT COUNT(*) as count FROM messages 
                WHERE user_id = ? AND timestamp >= ?
            """, (user_id, since_date.isoformat())).fetchone()['count']
            
            # Cost and token usage
            usage = conn.execute("""
                SELECT 
                    SUM(total_cost) as total_cost,
                    SUM(total_tokens) as total_tokens
                FROM conversations 
                WHERE user_id = ? AND created_at >= ?
            """, (user_id, since_date.isoformat())).fetchone()
            
            # Most used therapeutic areas
            therapeutic_areas = conn.execute("""
                SELECT therapeutic_area, COUNT(*) as count
                FROM conversations 
                WHERE user_id = ? AND created_at >= ? AND therapeutic_area IS NOT NULL
                GROUP BY therapeutic_area
                ORDER BY count DESC
                LIMIT 5
            """, (user_id, since_date.isoformat())).fetchall()
        
        return {
            "total_conversations": total_convs,
            "total_messages": total_msgs,
            "total_cost": usage['total_cost'] or 0.0,
            "total_tokens": usage['total_tokens'] or 0,
            "top_therapeutic_areas": [dict(row) for row in therapeutic_areas],
            "period_days": days
        }

# Global instance
chat_history_manager = ChatHistoryManager()

def get_chat_history_manager() -> ChatHistoryManager:
    """Get chat history manager instance"""
    return chat_history_manager
