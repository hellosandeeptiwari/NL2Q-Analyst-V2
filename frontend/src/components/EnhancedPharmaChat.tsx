import React, { useState, useEffect, useRef, useCallback } from 'react';
import './EnhancedPharmaChat.css';
import { 
  FiSend, FiCopy, FiCheck, FiDatabase, FiSettings, 
  FiBarChart2, FiTable, FiAlertTriangle, FiInfo, 
  FiChevronDown, FiChevronUp, FiLoader, FiUser,
  FiMessageSquare, FiStar, FiArchive, FiSearch,
  FiPlus, FiMoreHorizontal, FiDownload, FiShare2
} from 'react-icons/fi';
import Plot from 'react-plotly.js';

// Types
interface UserProfile {
  user_id: string;
  username: string;
  full_name: string;
  email: string;
  role: string;
  department: string;
  therapeutic_areas: string[];
  avatar_url?: string;
}

interface ChatMessage {
  message_id: string;
  conversation_id: string;
  message_type: 'user_query' | 'system_response' | 'sql_execution' | 'visualization' | 'error';
  content: string;
  timestamp: string;
  metadata?: any;
  status: 'pending' | 'processing' | 'completed' | 'error';
  response_time_ms?: number;
  cost_usd?: number;
}

interface Conversation {
  conversation_id: string;
  title: string;
  created_at: string;
  last_activity: string;
  therapeutic_area?: string;
  total_cost: number;
  total_tokens: number;
  is_favorite: boolean;
  is_archived: boolean;
  messages: ChatMessage[];
}

interface PlanStep {
  step_id: string;
  tool_type: string;
  status: 'pending' | 'executing' | 'completed' | 'error';
  start_time?: string;
  end_time?: string;
  error?: string;
}

interface QueryPlan {
  plan_id: string;
  status: 'draft' | 'validated' | 'executing' | 'completed' | 'failed';
  progress: number;
  current_step: string;
  reasoning_steps: string[];
  execution_steps: PlanStep[];
  estimated_cost: number;
  actual_cost: number;
  context?: any;
}

interface DatabaseStatus {
  isConnected: boolean;
  databaseType: string;
  server?: string;
  database?: string;
  schema?: string;
  warehouse?: string;
  lastConnected?: string;
}

// API Functions
const api = {
  // User Profile
  getUserProfile: async (userId: string): Promise<UserProfile> => {
    const response = await fetch(`/api/user/profile/${userId}`);
    return response.json();
  },

  // Chat History
  getConversations: async (userId: string, limit = 50): Promise<Conversation[]> => {
    const response = await fetch(`/api/chat/conversations/${userId}?limit=${limit}`);
    return response.json();
  },

  getConversation: async (conversationId: string): Promise<Conversation> => {
    const response = await fetch(`/api/chat/conversation/${conversationId}`);
    return response.json();
  },

  createConversation: async (userId: string, title?: string, therapeuticArea?: string): Promise<Conversation> => {
    const response = await fetch('/api/chat/conversation', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, title, therapeutic_area: therapeuticArea })
    });
    return response.json();
  },

  searchConversations: async (userId: string, query: string): Promise<Conversation[]> => {
    const response = await fetch(`/api/chat/search/${userId}?q=${encodeURIComponent(query)}`);
    return response.json();
  },

  // Database Status
  getDatabaseStatus: async (): Promise<DatabaseStatus> => {
    try {
      const response = await fetch('/api/database/status');
      if (!response.ok) {
        throw new Error('Failed to fetch database status');
      }
      return await response.json();
    } catch (error) {
      // Return default disconnected state if API fails
      return {
        isConnected: false,
        databaseType: 'Unknown',
        server: '',
        database: '',
        schema: '',
        warehouse: ''
      };
    }
  },

  // Refresh database status
  refreshDatabaseStatus: async (): Promise<DatabaseStatus> => {
    return api.getDatabaseStatus();
  },

  // Agent Query
  sendQuery: async (query: string, userId: string, conversationId: string): Promise<{ plan_id: string; status: string }> => {
    const response = await fetch('/api/agent/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        query, 
        user_id: userId, 
        session_id: conversationId 
      })
    });
    return response.json();
  },

  getPlanStatus: async (planId: string): Promise<QueryPlan> => {
    const response = await fetch(`/api/agent/plan/${planId}/status`);
    return response.json();
  },

  // Mock functions for development
  mockGetUserProfile: (): UserProfile => ({
    user_id: "user_123",
    username: "analyst1",
    full_name: "Dr. Sarah Chen",
    email: "sarah.chen@pharma.com",
    role: "Senior Data Analyst",
    department: "Commercial Analytics",
    therapeutic_areas: ["Oncology", "Diabetes", "Immunology"]
  }),

  mockGetConversations: (): Conversation[] => [
    {
      conversation_id: "conv_1",
      title: "Q4 Oncology Market Analysis",
      created_at: "2024-01-15T10:30:00Z",
      last_activity: "2024-01-15T15:45:00Z",
      therapeutic_area: "Oncology",
      total_cost: 2.45,
      total_tokens: 1250,
      is_favorite: true,
      is_archived: false,
      messages: []
    },
    {
      conversation_id: "conv_2", 
      title: "Physician Prescribing Patterns",
      created_at: "2024-01-14T09:15:00Z",
      last_activity: "2024-01-14T11:30:00Z",
      therapeutic_area: "Diabetes",
      total_cost: 1.85,
      total_tokens: 980,
      is_favorite: false,
      is_archived: false,
      messages: []
    }
  ]
};

interface EnhancedPharmaChatProps {
  onNavigateToSettings?: () => void;
}

const EnhancedPharmaChat: React.FC<EnhancedPharmaChatProps> = ({ onNavigateToSettings }) => {
  // State Management
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversation, setActiveConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activePlan, setActivePlan] = useState<QueryPlan | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showProfile, setShowProfile] = useState(false);
  const [databaseStatus, setDatabaseStatus] = useState<DatabaseStatus>({
    isConnected: false,
    databaseType: 'Unknown',
    server: '',
    database: '',
    schema: '',
    warehouse: ''
  });

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Scroll to bottom when messages change
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Initialize user data
  useEffect(() => {
    const initializeData = async () => {
      try {
        // Load user profile (using mock for now)
        const profile = api.mockGetUserProfile();
        setUserProfile(profile);

        // Load conversations (using mock for now)
        const convs = api.mockGetConversations();
        setConversations(convs);

        // Set first conversation as active
        if (convs.length > 0) {
          setActiveConversation(convs[0]);
          // In real implementation, load messages here
        }

        // Load database status
        const dbStatus = await api.getDatabaseStatus();
        setDatabaseStatus(dbStatus);
      } catch (error) {
        console.error('Error initializing data:', error);
      }
    };

    initializeData();
  }, []);

  // Refresh database status function
  const refreshDatabaseStatus = async () => {
    try {
      const dbStatus = await api.getDatabaseStatus();
      setDatabaseStatus(dbStatus);
    } catch (error) {
      console.error('Error refreshing database status:', error);
    }
  };

  // Add refresh when component comes into focus (e.g., returning from settings)
  useEffect(() => {
    const handleFocus = () => {
      refreshDatabaseStatus();
    };

    window.addEventListener('focus', handleFocus);
    return () => window.removeEventListener('focus', handleFocus);
  }, []);

  // Poll for plan status updates
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (activePlan && activePlan.status !== 'completed' && activePlan.status !== 'failed') {
      interval = setInterval(async () => {
        try {
          const statusRes = await api.getPlanStatus(activePlan.plan_id);
          setActivePlan(statusRes);
          
          if (statusRes.status === 'completed' || statusRes.status === 'failed') {
            setIsLoading(false);
          }
        } catch (error) {
          console.error('Error polling plan status:', error);
          setIsLoading(false);
        }
      }, 2000);
    }
    
    return () => clearInterval(interval);
  }, [activePlan]);

  // Handle sending messages
  const handleSendMessage = async () => {
    if (!currentMessage.trim() || !activeConversation || !userProfile) return;

    const userMessage: ChatMessage = {
      message_id: `msg_${Date.now()}`,
      conversation_id: activeConversation.conversation_id,
      message_type: 'user_query',
      content: currentMessage,
      timestamp: new Date().toISOString(),
      status: 'completed'
    };

    setMessages(prev => [...prev, userMessage]);
    setCurrentMessage('');
    setIsLoading(true);

    try {
      // Send query to agent
      const result = await api.sendQuery(currentMessage, userProfile.user_id, activeConversation.conversation_id);
      
      // Start polling for plan updates
      const planStatus = await api.getPlanStatus(result.plan_id);
      setActivePlan(planStatus);

    } catch (error) {
      console.error('Error sending message:', error);
      setIsLoading(false);
      
      const errorMessage: ChatMessage = {
        message_id: `msg_${Date.now()}`,
        conversation_id: activeConversation.conversation_id,
        message_type: 'error',
        content: 'Sorry, there was an error processing your request. Please try again.',
        timestamp: new Date().toISOString(),
        status: 'error'
      };
      
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  // Handle input key press
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Database Connection Indicator Component
  const DatabaseConnectionIndicator = () => {
    const getStatusColor = () => {
      return databaseStatus.isConnected ? '#10b981' : '#ef4444'; // green : red
    };

    const getStatusText = () => {
      if (!databaseStatus.isConnected) {
        return 'Not Connected';
      }
      
      let connectionText = databaseStatus.databaseType;
      if (databaseStatus.database) {
        connectionText += ` - ${databaseStatus.database}`;
      }
      if (databaseStatus.schema) {
        connectionText += `.${databaseStatus.schema}`;
      }
      return connectionText;
    };

    return (
      <div className="database-status-indicator" style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '4px 12px',
        backgroundColor: 'rgba(255, 255, 255, 0.1)',
        borderRadius: '20px',
        fontSize: '12px',
        fontWeight: '500'
      }}>
        <div 
          style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: getStatusColor(),
            boxShadow: databaseStatus.isConnected ? `0 0 6px ${getStatusColor()}` : 'none'
          }}
        />
        <FiDatabase size={12} />
        <span>{getStatusText()}</span>
      </div>
    );
  };

  // Create new conversation
  const handleNewConversation = async () => {
    if (!userProfile) return;

    try {
      const newConv = await api.createConversation(userProfile.user_id);
      setConversations(prev => [newConv, ...prev]);
      setActiveConversation(newConv);
      setMessages([]);
      setActivePlan(null);
    } catch (error) {
      console.error('Error creating conversation:', error);
    }
  };

  // Switch conversation
  const handleConversationSelect = (conversation: Conversation) => {
    setActiveConversation(conversation);
    setMessages(conversation.messages || []);
    setActivePlan(null);
  };

  // Quick action buttons
  const quickActions = [
    "Show Q4 oncology sales trends",
    "Analyze physician prescribing patterns", 
    "Compare therapeutic area performance",
    "Generate market share report",
    "Review patient adherence metrics"
  ];

  const handleQuickAction = (action: string) => {
    setCurrentMessage(action);
    inputRef.current?.focus();
  };

  // Format timestamp
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  // Render plan execution
  const renderPlanExecution = (plan: QueryPlan) => (
    <div className="plan-execution">
      <div className="plan-header">
        <h4 className="plan-title">AI Agent Execution Plan</h4>
        <span className={`plan-status ${plan.status}`}>
          {plan.status}
        </span>
      </div>
      
      {plan.reasoning_steps && plan.reasoning_steps.length > 0 && (
        <div className="reasoning-steps">
          <p className="reasoning-title">Reasoning:</p>
          <ul>
            {plan.reasoning_steps.map((step, idx) => (
              <li key={idx}>{step}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="plan-steps">
        {plan.execution_steps.map((step) => (
          <div key={step.step_id} className="plan-step">
            <div className={`step-status-icon ${step.status}`}>
              {step.status === 'completed' && '✓'}
              {step.status === 'executing' && <FiLoader />}
              {step.status === 'pending' && '○'}
              {step.status === 'error' && '!'}
            </div>
            <div className="step-details">
              <div className="step-name">{step.tool_type.replace('_', ' ')}</div>
              {step.error && (
                <div className="step-description error">{step.error}</div>
              )}
            </div>
          </div>
        ))}
      </div>

      {plan.estimated_cost > 0 && (
        <div className="plan-cost">
          Estimated cost: ${plan.estimated_cost.toFixed(3)}
        </div>
      )}
    </div>
  );

  // Render message
  const renderMessage = (message: ChatMessage) => (
    <div key={message.message_id} className={`message ${message.message_type === 'user_query' ? 'user' : 'assistant'}`}>
      <div className="message-avatar">
        {message.message_type === 'user_query' ? 
          (userProfile?.full_name.charAt(0) || 'U') : 'AI'
        }
      </div>
      <div className="message-content">
        <div className="message-bubble">
          {message.content}
        </div>
        <div className="message-metadata">
          <span>{formatTimestamp(message.timestamp)}</span>
          {message.cost_usd && (
            <span className="cost">${message.cost_usd.toFixed(3)}</span>
          )}
          {message.response_time_ms && (
            <span className="response-time">{message.response_time_ms}ms</span>
          )}
        </div>
      </div>
    </div>
  );

  if (!userProfile) {
    return <div className="loading-indicator">Loading...</div>;
  }

  return (
    <div className="pharma-chat-container">
      {/* Sidebar */}
      <div className="chat-sidebar">
        {/* Header */}
        <div className="sidebar-header">
          <h1>Pharma Analytics AI</h1>
          <p className="subtitle">Natural Language to Insights</p>
        </div>

        {/* User Profile */}
        <div className="user-profile-section">
          <div className="user-profile" onClick={() => setShowProfile(!showProfile)}>
            <div className="user-avatar">
              {userProfile.full_name.charAt(0)}
            </div>
            <div className="user-info">
              <h3>{userProfile.full_name}</h3>
              <p>{userProfile.role}</p>
            </div>
          </div>
          
          {userProfile.therapeutic_areas.length > 0 && (
            <div className="therapeutic-tags">
              {userProfile.therapeutic_areas.map(area => (
                <span key={area} className="therapeutic-tag">{area}</span>
              ))}
            </div>
          )}
        </div>

        {/* Chat History */}
        <div className="chat-history-section">
          <div className="chat-history-header">
            <h3>Recent Conversations</h3>
            <button className="new-chat-btn" onClick={handleNewConversation}>
              <FiPlus size={12} /> New
            </button>
          </div>

          <div className="chat-search">
            <input
              type="text"
              placeholder="Search conversations..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          <div className="conversation-list">
            {conversations
              .filter(conv => 
                conv.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                (conv.therapeutic_area && conv.therapeutic_area.toLowerCase().includes(searchQuery.toLowerCase()))
              )
              .map(conversation => (
                <div
                  key={conversation.conversation_id}
                  className={`conversation-item ${activeConversation?.conversation_id === conversation.conversation_id ? 'active' : ''}`}
                  onClick={() => handleConversationSelect(conversation)}
                >
                  <div className="conversation-title">
                    {conversation.is_favorite && <FiStar size={12} />}
                    {conversation.title}
                  </div>
                  <div className="conversation-meta">
                    <span className="conversation-date">
                      {formatTimestamp(conversation.last_activity)}
                    </span>
                    <span className="conversation-cost">
                      ${conversation.total_cost.toFixed(2)}
                    </span>
                  </div>
                  {conversation.therapeutic_area && (
                    <div className="therapeutic-tags">
                      <span className="therapeutic-tag">
                        {conversation.therapeutic_area}
                      </span>
                    </div>
                  )}
                </div>
              ))}
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="chat-main">
        {/* Chat Header */}
        <div className="chat-header">
          <div className="header-left">
            <h2 className="chat-title">
              {activeConversation?.title || 'New Conversation'}
            </h2>
            <DatabaseConnectionIndicator />
          </div>
          <div className="chat-actions">
            {onNavigateToSettings && (
              <button className="action-btn" onClick={onNavigateToSettings}>
                <FiSettings size={14} /> Settings
              </button>
            )}
            <button className="action-btn">
              <FiStar size={14} /> Favorite
            </button>
            <button className="action-btn">
              <FiDownload size={14} /> Export
            </button>
            <button className="action-btn">
              <FiShare2 size={14} /> Share
            </button>
            <button className="action-btn">
              <FiMoreHorizontal size={14} />
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="messages-container">
          {messages.map(renderMessage)}
          
          {activePlan && renderPlanExecution(activePlan)}
          
          {isLoading && (
            <div className="loading-indicator">
              <FiLoader className="loading-icon" />
              <span>AI is analyzing your request...</span>
              <div className="loading-dots">
                <div className="loading-dot"></div>
                <div className="loading-dot"></div>
                <div className="loading-dot"></div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="chat-input-container">
          {/* Quick Actions */}
          {messages.length === 0 && (
            <div className="quick-actions">
              {quickActions.map((action, idx) => (
                <button
                  key={idx}
                  className="quick-action"
                  onClick={() => handleQuickAction(action)}
                >
                  {action}
                </button>
              ))}
            </div>
          )}

          <div className="input-wrapper">
            <textarea
              ref={inputRef}
              className="chat-input"
              placeholder="Ask about your pharma data..."
              value={currentMessage}
              onChange={(e) => setCurrentMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isLoading}
              rows={1}
            />
            <button
              className="send-button"
              onClick={handleSendMessage}
              disabled={!currentMessage.trim() || isLoading}
            >
              <FiSend size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnhancedPharmaChat;
