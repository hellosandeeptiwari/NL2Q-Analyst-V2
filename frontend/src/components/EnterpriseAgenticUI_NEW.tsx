import React, { useState, useEffect, useRef } from 'react';
import './EnterpriseAgenticUI.css';

// Simple icon components to avoid TypeScript issues
const IconSend = ({ className }: { className?: string }) => <span className={className}>→</span>;
const IconCopy = ({ className }: { className?: string }) => <span className={className}>📋</span>;
const IconCheck = ({ className }: { className?: string }) => <span className={className}>✓</span>;
const IconDatabase = ({ className }: { className?: string }) => <span className={className}>🗄️</span>;
const IconSettings = ({ className }: { className?: string }) => <span className={className}>⚙️</span>;
const IconChart = ({ className }: { className?: string }) => <span className={className}>📊</span>;
const IconTable = ({ className }: { className?: string }) => <span className={className}>📋</span>;
const IconWarning = ({ className }: { className?: string }) => <span className={className}>⚠️</span>;
const IconInfo = ({ className }: { className?: string }) => <span className={className}>ℹ️</span>;
const IconDown = ({ className }: { className?: string }) => <span className={className}>▼</span>;
const IconUp = ({ className }: { className?: string }) => <span className={className}>▲</span>;
const IconLoader = ({ className }: { className?: string }) => <span className={`spin ${className || ''}`}>⟳</span>;
const IconPlus = ({ className }: { className?: string }) => <span className={className}>+</span>;
const IconEdit = ({ className }: { className?: string }) => <span className={className}>✎</span>;
const IconDelete = ({ className }: { className?: string }) => <span className={className}>🗑️</span>;
const IconUser = ({ className }: { className?: string }) => <span className={className}>👤</span>;
const IconLogout = ({ className }: { className?: string }) => <span className={className}>↩</span>;
const IconMenu = ({ className }: { className?: string }) => <span className={className}>☰</span>;

// Type definitions
interface Message {
  id: string;
  type: 'user' | 'agent' | 'error';
  text?: string;
  plan?: any;
  timestamp: Date;
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  lastActivity: Date;
  isArchived?: boolean;
}

interface User {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  role: string;
  subscriptionPlan: string;
}

interface ActivePlan {
  plan_id: string;
  status: string;
}

// Mock conversational responses
const conversationalResponses = {
  greetings: [
    "Hello! I'm your Enterprise NL2Query Agent. I can help you with data analysis, SQL queries, and business intelligence. What would you like to explore today?",
    "Hi there! Ready to dive into some data insights? I can help you analyze your data, create visualizations, and answer complex business questions.",
    "Welcome back! I'm here to help you unlock insights from your data. Feel free to ask me anything about your datasets or business metrics."
  ],
  
  howAreYou: [
    "I'm doing great and ready to help you with your data analysis needs! How can I assist you today?",
    "I'm functioning perfectly and excited to help you discover insights in your data. What would you like to analyze?",
    "I'm excellent, thank you for asking! I'm here to help you with any data queries or analysis you need."
  ],
  
  thanks: [
    "You're welcome! I'm always here to help with your data analysis needs.",
    "My pleasure! Feel free to ask if you need anything else.",
    "Happy to help! Let me know if you have any other questions."
  ],
  
  capabilities: [
    "I can help you with data analysis, SQL query generation, creating visualizations, cost estimation, and providing business insights. I use an advanced agentic approach for complex data tasks.",
    "My capabilities include natural language to SQL conversion, data visualization, schema discovery, query optimization, cost estimation, and comprehensive data analysis with governance controls.",
    "I can analyze your data, generate SQL queries, create charts and reports, estimate costs, cache results for performance, and ensure data governance compliance. What specific task interests you?"
  ]
};

// Mock API functions
const mockApi = {
  agentQuery: async (query: string, userId: string, sessionId: string) => {
    console.log("Sending query:", { query, userId, sessionId });
    
    // Check if it's a conversational query
    const lowerQuery = query.toLowerCase();
    if (lowerQuery.includes('hello') || lowerQuery.includes('hi') || lowerQuery.includes('hey')) {
      return { 
        type: 'conversational', 
        response: conversationalResponses.greetings[Math.floor(Math.random() * conversationalResponses.greetings.length)]
      };
    }
    
    if (lowerQuery.includes('how are you') || lowerQuery.includes('how do you do')) {
      return { 
        type: 'conversational', 
        response: conversationalResponses.howAreYou[Math.floor(Math.random() * conversationalResponses.howAreYou.length)]
      };
    }
    
    if (lowerQuery.includes('thank') || lowerQuery.includes('thanks')) {
      return { 
        type: 'conversational', 
        response: conversationalResponses.thanks[Math.floor(Math.random() * conversationalResponses.thanks.length)]
      };
    }
    
    if (lowerQuery.includes('what can you do') || lowerQuery.includes('capabilities') || lowerQuery.includes('help')) {
      return { 
        type: 'conversational', 
        response: conversationalResponses.capabilities[Math.floor(Math.random() * conversationalResponses.capabilities.length)]
      };
    }
    
    // For data queries, trigger agentic approach
    await new Promise(res => setTimeout(res, 1000));
    const planId = `plan_${Date.now()}`;
    return { 
      type: 'agentic',
      plan_id: planId, 
      status: 'draft' 
    };
  },
  
  getPlanStatus: async (planId: string) => {
    console.log("Getting status for plan:", planId);
    const statuses = ['draft', 'validated', 'executing', 'completed'];
    const currentStatus = statuses[Math.floor(Math.random() * statuses.length)];
    const progress = Math.random();
    
    let output = null;
    if (currentStatus === 'completed') {
      output = {
        visualizations: {
          charts: [
            { type: 'bar', title: 'Sales by Region', config: { x_column: 'Region', y_column: 'Sales' } }
          ]
        },
        query_results: {
          data: [
            { Region: 'North', Sales: 1000 },
            { Region: 'South', Sales: 1500 },
            { Region: 'East', Sales: 1200 },
            { Region: 'West', Sales: 1800 },
          ]
        }
      };
    }

    return {
      plan_id: planId,
      status: currentStatus,
      progress: progress,
      current_step: 'sql_generation',
      estimated_cost: 1.25,
      actual_cost: 0.95,
      reasoning_steps: ["Identified entities: sales, region", "Aggregation: sum of sales", "Grouping by: region"],
      execution_steps: [
        { step_id: '1', tool_type: 'SCHEMA_DISCOVERY', status: 'completed' },
        { step_id: '2', tool_type: 'SQL_GENERATION', status: 'executing' },
        { step_id: '3', tool_type: 'EXECUTION', status: 'pending' },
      ],
      context: {
        generated_sql: "SELECT region, SUM(sales) as total_sales FROM sales_table GROUP BY region;",
        ...output
      }
    };
  },
  
  approvePlan: async (planId: string, approverId: string) => {
    console.log("Approving plan:", { planId, approverId });
    await new Promise(res => setTimeout(res, 500));
    return { status: 'approved', plan_id: planId };
  },

  saveChatSession: async (session: ChatSession) => {
    localStorage.setItem(`chat_session_${session.id}`, JSON.stringify(session));
    return session;
  },

  loadChatSessions: async (): Promise<ChatSession[]> => {
    const sessions: ChatSession[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key?.startsWith('chat_session_')) {
        const sessionData = localStorage.getItem(key);
        if (sessionData) {
          const session = JSON.parse(sessionData);
          // Parse dates
          session.lastActivity = new Date(session.lastActivity);
          session.messages = session.messages.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp)
          }));
          sessions.push(session);
        }
      }
    }
    return sessions.sort((a, b) => new Date(b.lastActivity).getTime() - new Date(a.lastActivity).getTime());
  },

  deleteChatSession: async (sessionId: string) => {
    localStorage.removeItem(`chat_session_${sessionId}`);
  }
};

const EnterpriseAgenticUI = () => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activePlan, setActivePlan] = useState<ActivePlan | null>(null);
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Mock user data
  const currentUser: User = {
    id: 'user_123',
    name: 'John Doe',
    email: 'john.doe@company.com',
    role: 'Data Analyst',
    subscriptionPlan: 'Enterprise Pro'
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  useEffect(() => {
    loadChatSessions();
  }, []);

  useEffect(() => {
    if (currentSession && messages.length > 0) {
      const updatedSession = {
        ...currentSession,
        messages: messages,
        lastActivity: new Date()
      };
      setCurrentSession(updatedSession);
      mockApi.saveChatSession(updatedSession);
      loadChatSessions();
    }
  }, [messages]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (activePlan && activePlan.status !== 'completed' && activePlan.status !== 'failed') {
      interval = setInterval(async () => {
        const statusRes = await mockApi.getPlanStatus(activePlan.plan_id);
        
        setMessages(prev => prev.map(msg => 
          msg.id === activePlan.plan_id ? { ...msg, plan: statusRes } : msg
        ));

        if (statusRes.status === 'completed' || statusRes.status === 'failed') {
          setActivePlan(null);
          clearInterval(interval);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [activePlan]);

  const loadChatSessions = async () => {
    const sessions = await mockApi.loadChatSessions();
    setChatSessions(sessions);
  };

  const createNewChat = () => {
    const newSession: ChatSession = {
      id: `session_${Date.now()}`,
      title: 'New Chat',
      messages: [],
      lastActivity: new Date()
    };
    setCurrentSession(newSession);
    setMessages([]);
    setActivePlan(null);
  };

  const loadChatSession = (session: ChatSession) => {
    setCurrentSession(session);
    setMessages(session.messages);
    setActivePlan(null);
  };

  const deleteChatSession = async (sessionId: string) => {
    await mockApi.deleteChatSession(sessionId);
    if (currentSession?.id === sessionId) {
      createNewChat();
    }
    loadChatSessions();
  };

  const updateSessionTitle = (sessionId: string, newTitle: string) => {
    setChatSessions(prev => prev.map(session => 
      session.id === sessionId ? { ...session, title: newTitle } : session
    ));
  };

  const handleSendQuery = async () => {
    if (!query.trim()) return;

    if (!currentSession) {
      createNewChat();
    }

    const userMessage: Message = {
      id: `user_${Date.now()}`,
      type: 'user',
      text: query,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    
    // Auto-generate title for new sessions
    if (currentSession && currentSession.messages.length === 0) {
      const title = query.length > 30 ? query.substring(0, 30) + '...' : query;
      const updatedSession = { ...currentSession, title };
      setCurrentSession(updatedSession);
      updateSessionTitle(currentSession.id, title);
    }
    
    setQuery('');
    setIsLoading(true);

    try {
      const res = await mockApi.agentQuery(query, currentUser.id, currentSession?.id || '');
      
      if (res.type === 'conversational') {
        // Handle conversational response
        const agentMessage: Message = {
          id: `agent_${Date.now()}`,
          type: 'agent',
          text: res.response,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, agentMessage]);
      } else if (res.type === 'agentic' && res.plan_id) {
        // Handle agentic response with plan
        const agentMessage: Message = {
          id: res.plan_id,
          type: 'agent',
          plan: {
            plan_id: res.plan_id,
            status: 'draft',
            progress: 0,
            reasoning_steps: [],
            execution_steps: [],
            context: {}
          },
          timestamp: new Date()
        };
        setMessages(prev => [...prev, agentMessage]);
        setActivePlan({ plan_id: res.plan_id, status: 'draft' });
      }
    } catch (error) {
      const errorMessage: Message = {
        id: `error_${Date.now()}`,
        type: 'error',
        text: 'Sorry, I encountered an error while processing your request. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const approvePlan = async (planId: string) => {
    try {
      await mockApi.approvePlan(planId, currentUser.id);
      setActivePlan({ plan_id: planId, status: 'approved' });
    } catch (error) {
      console.error('Failed to approve plan:', error);
    }
  };

  const formatTime = (date: Date) => {
    return new Date(date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const formatDate = (date: Date) => {
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - new Date(date).getTime()) / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    return new Date(date).toLocaleDateString();
  };

  const renderMessage = (message: Message) => {
    const isUser = message.type === 'user';
    const isError = message.type === 'error';

    return (
      <div key={message.id} className={`message ${isUser ? 'user-message' : isError ? 'error-message' : 'agent-message'}`}>
        <div className="message-header">
          <span className="message-sender">
            {isUser ? currentUser.name : isError ? 'System' : 'Enterprise Agent'}
          </span>
          <span className="message-time">{formatTime(message.timestamp)}</span>
        </div>
        
        {message.text && (
          <div className="message-content">
            {message.text}
          </div>
        )}

        {message.plan && (
          <div className="plan-execution">
            <div className="plan-header">
              <div className="plan-status">
                <IconCheck className={`status-icon ${message.plan.status}`} />
                <span>Status: {message.plan.status}</span>
              </div>
              <div className="plan-progress">
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${(message.plan.progress || 0) * 100}%` }}
                  ></div>
                </div>
                <span className="progress-text">
                  {Math.round((message.plan.progress || 0) * 100)}%
                </span>
              </div>
            </div>

            {message.plan.estimated_cost && (
              <div className="cost-info">
                <span>Est. Cost: ${message.plan.estimated_cost}</span>
                {message.plan.actual_cost && (
                  <span>Actual Cost: ${message.plan.actual_cost}</span>
                )}
              </div>
            )}

            {message.plan.context?.visualizations && (
              <div className="results-section">
                <div className="result-tabs">
                  <button className="tab active">
                    <IconChart className="tab-icon" />
                    Chart
                  </button>
                  <button className="tab">
                    <IconTable className="tab-icon" />
                    Table
                  </button>
                </div>
                
                <div className="chart-placeholder">
                  <h4>{message.plan.context.visualizations.charts[0]?.title}</h4>
                  <div className="chart-info">
                    <span>Chart Type: {message.plan.context.visualizations.charts[0]?.type}</span>
                    <br />
                    <span>X: {message.plan.context.visualizations.charts[0]?.config.x_column}, Y: {message.plan.context.visualizations.charts[0]?.config.y_column}</span>
                  </div>
                </div>
              </div>
            )}

            {message.plan.status === 'draft' && (
              <div className="plan-actions">
                <button 
                  className="approve-btn"
                  onClick={() => approvePlan(message.plan.plan_id)}
                >
                  Approve & Execute
                </button>
              </div>
            )}

            <details className="execution-details">
              <summary>
                <IconDown className="expand-icon" />
                Execution Details
              </summary>
              <div className="details-content">
                {message.plan.reasoning_steps?.length > 0 && (
                  <div className="reasoning-steps">
                    <h4>Reasoning:</h4>
                    <ul>
                      {message.plan.reasoning_steps.map((step: string, idx: number) => (
                        <li key={idx}>{step}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {message.plan.execution_steps?.length > 0 && (
                  <div className="execution-steps">
                    <h4>Execution Steps:</h4>
                    {message.plan.execution_steps.map((step: any) => (
                      <div key={step.step_id} className={`execution-step ${step.status}`}>
                        <IconCheck className="step-icon" />
                        <span>{step.tool_type}: {step.status}</span>
                      </div>
                    ))}
                  </div>
                )}

                {message.plan.context?.generated_sql && (
                  <div className="sql-preview">
                    <h4>Generated SQL:</h4>
                    <pre>{message.plan.context.generated_sql}</pre>
                  </div>
                )}
              </div>
            </details>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="enterprise-agentic-ui">
      {/* Sidebar */}
      <div className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <button className="new-chat-btn" onClick={createNewChat}>
            <IconPlus />
            New Chat
          </button>
        </div>
        
        <div className="chat-history">
          {chatSessions.map(session => (
            <div 
              key={session.id} 
              className={`chat-item ${currentSession?.id === session.id ? 'active' : ''}`}
              onClick={() => loadChatSession(session)}
            >
              <div className="chat-title">{session.title}</div>
              <div className="chat-meta">
                <span className="chat-date">{formatDate(session.lastActivity)}</span>
                <button 
                  className="delete-chat"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteChatSession(session.id);
                  }}
                >
                  <IconDelete />
                </button>
              </div>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <div className="user-section">
            <div 
              className="user-info"
              onClick={() => setUserMenuOpen(!userMenuOpen)}
            >
              <IconUser className="user-avatar" />
              <div className="user-details">
                <div className="user-name">{currentUser.name}</div>
                <div className="user-plan">{currentUser.subscriptionPlan}</div>
              </div>
            </div>
            
            {userMenuOpen && (
              <div className="user-menu">
                <div className="user-menu-item">
                  <IconSettings />
                  Settings
                </div>
                <div className="user-menu-item">
                  <IconLogout />
                  Sign Out
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className="header">
          <button 
            className="sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            <IconMenu />
          </button>
          <h1>Enterprise NL2Query Agent</h1>
          <div className="connection-status">
            <div className="status-indicator connected"></div>
            <span>Connected</span>
          </div>
        </div>

        <div className="messages-container">
          {messages.length === 0 && (
            <div className="welcome-message">
              <h2>Welcome to Enterprise NL2Query Agent</h2>
              <p>I'm your intelligent data analysis assistant. I can help you with:</p>
              <ul>
                <li>Natural language to SQL conversion</li>
                <li>Data visualization and charts</li>
                <li>Business intelligence insights</li>
                <li>Query optimization and cost estimation</li>
                <li>Data governance and compliance</li>
              </ul>
              <p>Start a conversation or ask me to analyze your data!</p>
            </div>
          )}
          
          {messages.map(renderMessage)}
          {isLoading && (
            <div className="typing-indicator">
              <IconLoader className="loader" />
              <span>Agent is thinking...</span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <div className="query-input">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSendQuery()}
              placeholder="Ask a question about your data..."
              disabled={isLoading}
            />
            <button 
              onClick={handleSendQuery}
              disabled={isLoading || !query.trim()}
              className="send-button"
            >
              <IconSend />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnterpriseAgenticUI;
