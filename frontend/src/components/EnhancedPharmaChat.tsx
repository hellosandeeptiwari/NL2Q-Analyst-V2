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
  input_data?: any;
  output_data?: {
    sql?: string;
    results?: any[];
    chart_config?: {
      chart_type?: string;
      [key: string]: any;
    };
    [key: string]: any;
  };
}

interface QueryPlan {
  plan_id: string;
  status: 'draft' | 'validated' | 'executing' | 'completed' | 'failed';
  progress: number;
  current_step: string;
  reasoning_steps: string[];
  tasks: PlanStep[];  // Backend returns 'tasks', not 'execution_steps'
  results: { [key: string]: any };  // Backend returns results object
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
      const response = await fetch('http://localhost:8000/api/database/status');
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

  // Agent Query (no timeout - waits indefinitely for backend)
  sendQuery: async (query: string, userId: string, conversationId: string): Promise<QueryPlan> => {
    // No timeout - let the backend take as long as it needs
    const response = await fetch('http://localhost:8000/api/agent/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        query, 
        user_id: userId, 
        session_id: conversationId 
      })
      // Explicitly no timeout specified - browser default is usually 5+ minutes
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  },

  // Intent Detection - Check if message needs planning or casual response
  detectIntent: async (query: string): Promise<{needsPlanning: boolean, response?: string}> => {
    const response = await fetch('http://localhost:8000/api/agent/detect-intent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  },

  getPlanStatus: async (planId: string): Promise<QueryPlan> => {
    // No timeout for status checks either
    const response = await fetch(`http://localhost:8000/api/agent/plan/${planId}/status`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  },

  // Mock functions for development
  mockGetUserProfile: (): UserProfile => ({
    user_id: "default_user",
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
  const [expandedSteps, setExpandedSteps] = useState<{[key: string]: boolean}>({});
  const [showStepsDetails, setShowStepsDetails] = useState(false);
  const [selectedStepKey, setSelectedStepKey] = useState<string | null>(null);
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

  // Toggle step expansion
  const toggleStepExpansion = (messageId: string) => {
    setExpandedSteps(prev => ({
      ...prev,
      [messageId]: !prev[messageId]
    }));
  };

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

  // Poll for plan status updates - DISABLED since we get complete plan directly
  useEffect(() => {
    // No longer needed - backend returns complete plan immediately
    // The plan status polling was causing undefined plan_id errors
    
    // Just ensure loading state is turned off when plan is set
    if (activePlan && (activePlan.status === 'completed' || activePlan.status === 'failed')) {
      setIsLoading(false);
    }
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
    const messageContent = currentMessage;
    setCurrentMessage('');
    setIsLoading(true);

    try {
      // First, detect intent to see if this needs planning or just casual response
      console.log('Detecting intent for message:', messageContent);
      
      let intentResult;
      try {
        intentResult = await api.detectIntent(messageContent);
        console.log('Intent detection result:', intentResult);
      } catch (intentError) {
        console.warn('Intent detection failed, falling back to planning:', intentError);
        // If intent detection fails, default to planning (backward compatibility)
        intentResult = { needsPlanning: true };
      }

      if (!intentResult.needsPlanning) {
        // Handle as casual conversation
        console.log('Treating as casual conversation');
        setIsLoading(false);
        
        const assistantMessage: ChatMessage = {
          message_id: `msg_${Date.now()}_assistant`,
          conversation_id: activeConversation.conversation_id,
          message_type: 'system_response',
          content: intentResult.response || 'Hello! I\'m here to help with your data analysis needs. Feel free to ask me about your pharmaceutical data!',
          timestamp: new Date().toISOString(),
          status: 'completed'
        };
        
        setMessages(prev => [...prev, assistantMessage]);
        return;
      }

      // If planning is needed, proceed with the full workflow
      console.log('Intent requires planning, sending to agent...');
      const planResult = await api.sendQuery(messageContent, userProfile.user_id, activeConversation.conversation_id);
      
      // Add debugging to see what we're getting
      console.log('Received plan result:', planResult);
      console.log('Plan status:', planResult.status);
      console.log('Plan tasks:', planResult.tasks);
      
      // Set the plan directly since it's already the full plan object
      setActivePlan(planResult);
      
      // If plan is completed, add assistant response to messages and clear active plan
      if (planResult.status === 'completed' || planResult.status === 'failed') {
        setIsLoading(false);
        
        // Create assistant response message
        const assistantMessage: ChatMessage = {
          message_id: `msg_${Date.now()}_assistant`,
          conversation_id: activeConversation.conversation_id,
          message_type: 'system_response',
          content: planResult.status === 'completed' ? 
            'Analysis completed successfully. See results below.' : 
            'Analysis completed with some issues. See details below.',
          timestamp: new Date().toISOString(),
          status: planResult.status === 'failed' ? 'error' : planResult.status
        };
        
        // Add assistant message and clear active plan after a short delay
        setTimeout(() => {
          setMessages(prev => [...prev, assistantMessage]);
          // Keep the plan visible for results, but don't show it in the main flow
          // setActivePlan(null);
        }, 500);
      } else {
        // Plan is still executing, start polling for updates
        console.log('Plan is executing, starting to poll for status...');
        let pollAttempts = 0;
        
        const pollPlanStatus = async () => {
          pollAttempts++;
          console.log(`Polling attempt ${pollAttempts} for plan ${planResult.plan_id}`);
          
          try {
            const updatedPlan = await api.getPlanStatus(planResult.plan_id);
            console.log('Polling plan status:', updatedPlan.status, `Progress: ${updatedPlan.progress}%`);
            setActivePlan(updatedPlan);
            
            if (updatedPlan.status === 'completed' || updatedPlan.status === 'failed') {
              console.log(`Plan ${updatedPlan.status} after ${pollAttempts} polling attempts`);
              setIsLoading(false);
              
              const assistantMessage: ChatMessage = {
                message_id: `msg_${Date.now()}_assistant`,
                conversation_id: activeConversation.conversation_id,
                message_type: 'system_response',
                content: updatedPlan.status === 'completed' ? 
                  'Analysis completed successfully. See results below.' : 
                  'Analysis completed with some issues. See details below.',
                timestamp: new Date().toISOString(),
                status: updatedPlan.status === 'failed' ? 'error' : updatedPlan.status
              };
              
              setTimeout(() => {
                setMessages(prev => [...prev, assistantMessage]);
              }, 500);
            } else {
              // Continue polling - will wait indefinitely until completion
              console.log(`Plan still ${updatedPlan.status}, continuing to poll...`);
              setTimeout(pollPlanStatus, 1000);
            }
          } catch (error) {
            console.error('Error polling plan status:', error);
            pollAttempts++;
            
            // Don't give up on errors - retry polling after a longer delay
            if (pollAttempts < 1000) { // Effectively infinite retries
              console.log('Retrying polling after error...');
              setTimeout(pollPlanStatus, 3000); // Wait 3 seconds after error
            } else {
              console.log('Too many polling errors, stopping');
              setIsLoading(false);
            }
          }
        };
        
        // Start polling after 1 second
        setTimeout(pollPlanStatus, 1000);
      }

    } catch (error) {
      console.error('Error sending message:', error);
      setIsLoading(false);
      
      // Show detailed error instead of generic message
      let errorDetails = 'Unknown error occurred';
      if (error instanceof Error) {
        errorDetails = error.message;
      } else if (typeof error === 'string') {
        errorDetails = error;
      }
      
      const errorMessage: ChatMessage = {
        message_id: `msg_${Date.now()}`,
        conversation_id: activeConversation.conversation_id,
        message_type: 'error',
        content: `Error processing request: ${errorDetails}. Please check the console for more details and try again.`,
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
    if (!userProfile) {
      console.error('No user profile available for creating conversation');
      return;
    }

    try {
      setIsLoading(true);
      const newConv = await api.createConversation(userProfile.user_id);
      
      // Validate the response
      if (!newConv || !newConv.conversation_id) {
        throw new Error('Invalid conversation response from server');
      }
      
      setConversations(prev => [newConv, ...prev]);
      setActiveConversation(newConv);
      setMessages([]);
      setActivePlan(null);
      setSelectedStepKey(null);
      setShowStepsDetails(false);
      
      console.log('New conversation created successfully:', newConv.conversation_id);
    } catch (error) {
      console.error('Error creating conversation:', error);
      
      // Create a fallback local conversation if API fails
      const fallbackConv: Conversation = {
        conversation_id: `local_${Date.now()}`,
        title: 'New Chat',
        created_at: new Date().toISOString(),
        last_activity: new Date().toISOString(),
        total_cost: 0,
        total_tokens: 0,
        is_favorite: false,
        is_archived: false,
        messages: []
      };
      
      setConversations(prev => [fallbackConv, ...prev]);
      setActiveConversation(fallbackConv);
      setMessages([]);
      setActivePlan(null);
      setSelectedStepKey(null);
      setShowStepsDetails(false);
      
      console.log('Created fallback conversation:', fallbackConv.conversation_id);
    } finally {
      setIsLoading(false);
    }
  };

  // Switch conversation
  const handleConversationSelect = (conversation: Conversation) => {
    setActiveConversation(conversation);
    // Sort messages by timestamp to ensure correct order
    const sortedMessages = (conversation.messages || []).sort((a, b) => 
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );
    setMessages(sortedMessages);
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

  // Render plan execution with live status
  const renderPlanExecution = (plan: QueryPlan) => {
    console.log('Plan tasks:', plan.tasks);
    console.log('Plan results:', plan.results);
    console.log('Plan current step:', plan.current_step);
    console.log('Plan progress:', plan.progress);
    
    // Define the expected steps in order - using actual planner task IDs
    const expectedSteps = [
      { key: '1_discover_schema', name: 'Database Discovery', description: 'Analyzing database structure and available tables' },
      { key: '2_semantic_analysis', name: 'Query Understanding', description: 'Interpreting your natural language request' },
      { key: '3_similarity_matching', name: 'Table Matching', description: 'Finding the most relevant tables for your query' },
      { key: '4_user_verification', name: 'Table Selection', description: 'Selecting the best tables for your analysis' },
      { key: '5_query_generation', name: 'SQL Generation', description: 'Creating optimized database query' },
      { key: '6_query_execution', name: 'Data Retrieval', description: 'Executing query and fetching your results' },
      { key: '7_visualization', name: 'Chart Creation', description: 'Generating visualizations from your data' }
    ];
    
    // Calculate current step index
    const currentStepIndex = plan.status === 'completed' ? expectedSteps.length : 
      (plan.current_step ? expectedSteps.findIndex(step => step.key === plan.current_step) : -1);
    
    // Separate intermediate steps (1-6) from final display step (7)
    const intermediateTasks = plan.tasks ? plan.tasks.filter((task, index) => index < plan.tasks.length - 1) : [];
    const finalTask = plan.tasks && plan.tasks.length > 0 ? plan.tasks[plan.tasks.length - 1] : null;
    
    // Handle undefined status gracefully
    const planStatus = plan.status || 'unknown';
    const isExecuting = planStatus === 'executing' || planStatus === 'draft' || planStatus === 'validated' || planStatus === 'completed';
    const progressPercentage = Math.round((plan.progress || 0) * 100);
    
    return (
    <div className="plan-execution">
      {/* Horizontal Steps Progress Indicator */}
      {plan && (
        <div className="horizontal-steps-container">
          {/* Toggle Button */}
          <div className="steps-toggle-header" onClick={() => setShowStepsDetails(!showStepsDetails)}>
            <span className="steps-toggle-title">
              🤖 AI Agent Execution Plan 
              <span className="steps-status">
                {plan.status === 'completed' ? 'COMPLETED' : 'IN PROGRESS'}
              </span>
            </span>
            <FiChevronDown className={`toggle-icon ${showStepsDetails ? 'rotated' : ''}`} />
          </div>

          {/* Horizontal Steps Timeline - Dynamic Display */}
          <div className="horizontal-steps-timeline">
            {expectedSteps.map((step, index) => {
              const isCompleted = currentStepIndex > index;
              const isCurrent = currentStepIndex === index;
              
              // For completed plans, show all steps. For executing plans, show current and completed
              const shouldShow = plan.status === 'completed' ? true : (isCompleted || isCurrent);
              
              if (!shouldShow) return null;
              
              return (
                <div 
                  key={step.key} 
                  className={`horizontal-step-item ${isCompleted ? 'completed' : isCurrent ? 'current' : 'pending'} ${selectedStepKey === step.key ? 'selected' : ''} dynamic-reveal`}
                  onClick={() => setSelectedStepKey(selectedStepKey === step.key ? null : step.key)}
                >
                  <div className="horizontal-step-circle">
                    {isCompleted ? (
                      <div className="step-checkmark">✓</div>
                    ) : isCurrent ? (
                      <div className="step-spinner">⚙️</div>
                    ) : (
                      <div className="step-number">{index + 1}</div>
                    )}
                  </div>
                  <div className="horizontal-step-label">{step.name}</div>
                  
                  {/* Show connector only if next step is also visible */}
                  {index < expectedSteps.length - 1 && currentStepIndex > index && (
                    <div className={`step-connector ${isCompleted ? 'completed' : ''}`}></div>
                  )}
                </div>
              );
            })}
            
            {/* Show remaining steps count - only for executing plans */}
            {plan.status !== 'completed' && currentStepIndex < expectedSteps.length - 1 && (
              <div className="remaining-steps-indicator">
                <div className="remaining-steps-circle">
                  <span className="remaining-count">+{expectedSteps.length - currentStepIndex - 1}</span>
                </div>
                <div className="remaining-steps-label">More steps</div>
              </div>
            )}
          </div>

          {/* Step Details Panel - Only for visible steps */}
          {showStepsDetails && selectedStepKey && (
            <div className="step-details-panel">
              {(() => {
                const selectedStep = expectedSteps.find(s => s.key === selectedStepKey);
                const stepIndex = expectedSteps.findIndex(s => s.key === selectedStepKey);
                const isCompleted = currentStepIndex > stepIndex;
                const isCurrent = currentStepIndex === stepIndex;
                
                return (
                  <div className="step-detail-content">
                    <div className="step-detail-header">
                      <h4>{selectedStep?.name}</h4>
                      <span className={`step-detail-status ${isCompleted ? 'completed' : isCurrent ? 'current' : 'pending'}`}>
                        {isCompleted ? 'Completed' : isCurrent ? 'In Progress' : 'Pending'}
                      </span>
                    </div>
                    <p className="step-detail-description">{selectedStep?.description}</p>
                    
                    {/* Show specific details based on step - only for completed steps */}
                    {isCompleted && selectedStepKey === '1_discover_schema' && plan.results?.['1_discover_schema'] && (
                      <div className="step-result-details">
                        <p>✅ Database schema discovered successfully</p>
                        <p>📊 Tables and columns identified for analysis</p>
                      </div>
                    )}
                    
                    {isCompleted && selectedStepKey === '2_semantic_analysis' && plan.results?.['2_semantic_analysis'] && (
                      <div className="step-result-details">
                        <p>🧠 Query intent understood</p>
                        <p>🔍 Key entities extracted and analyzed</p>
                      </div>
                    )}
                    
                    {isCompleted && selectedStepKey === '3_similarity_matching' && plan.results?.['3_similarity_matching'] && (
                      <div className="step-result-details">
                        <p>🎯 Matching tables identified successfully</p>
                        <p>📈 Relevance scores calculated</p>
                      </div>
                    )}
                    
                    {isCompleted && selectedStepKey === '5_query_generation' && plan.results?.['5_query_generation'] && (
                      <div className="step-result-details">
                        <p>✅ SQL query generated successfully</p>
                        {plan.results['5_query_generation'].sql_query && (
                          <div className="sql-preview">
                            <code>{plan.results['5_query_generation'].sql_query.substring(0, 100)}...</code>
                          </div>
                        )}
                      </div>
                    )}
                    
                    {isCompleted && selectedStepKey === '6_query_execution' && plan.results?.['6_query_execution'] && (
                      <div className="step-result-details">
                        <p>✅ Query executed successfully</p>
                        <p>📊 Data retrieved and processed</p>
                      </div>
                    )}
                    
                    {isCompleted && selectedStepKey === '7_visualization' && plan.results?.['7_visualization'] && (
                      <div className="step-result-details">
                        <p>📈 Visualizations created successfully</p>
                        <p>🎨 Charts ready for display</p>
                      </div>
                    )}
                    
                    {/* Show current step progress */}
                    {isCurrent && (
                      <div className="step-progress-details">
                        <div className="progress-indicator">
                          <div className="progress-bar">
                            <div className="progress-fill"></div>
                          </div>
                          <p className="progress-text">Processing step...</p>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      )}
      
      {/* Enhanced Reasoning Section - HIDDEN */}
      {false && plan.reasoning_steps && plan.reasoning_steps.length > 0 && (
        <div className="reasoning-steps">
          <div className="reasoning-header">
            <strong>🧠 Reasoning Process:</strong>
          </div>
          <ul className="reasoning-list">
            {plan.reasoning_steps.map((step, idx) => (
              <li key={idx} className="reasoning-item">
                <span className="reasoning-bullet">•</span>
                <span className="reasoning-text">{step}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Collapsible Intermediate Steps (1-6) - HIDDEN FOR NEW HORIZONTAL UI */}
      {false && intermediateTasks.length > 0 && (
        <div className="plan-steps">
          <div className="steps-header" onClick={() => toggleStepExpansion(plan.plan_id)}>
            <p className="steps-title">
              <strong>📋 Processing Details ({intermediateTasks.length} steps)</strong>
              {!expandedSteps[plan.plan_id] && (
                <span className="steps-summary">
                  {' '}- {plan.status === 'completed' ? '✅ All steps completed' : 
                        plan.status === 'failed' ? '❌ Execution failed' :
                        isExecuting ? `⚙️ ${progressPercentage}% complete` : '⏳ Pending'}
                </span>
              )}
            </p>
            <span className={`steps-toggle ${expandedSteps[plan.plan_id] ? 'expanded' : 'collapsed'}`}>
              {expandedSteps[plan.plan_id] ? '▼' : '▶'}
            </span>
          </div>
          
          {expandedSteps[plan.plan_id] && (
            <div className="steps-content">
              {intermediateTasks.map((task: any, index: number) => {
          // Fix the result key mapping to match backend format
          const taskTypeMap: { [key: string]: string } = {
            'schema_discovery': 'discover_schema',
            'semantic_understanding': 'semantic_analysis', 
            'similarity_matching': 'similarity_matching',
            'user_interaction': 'user_verification',
            'query_generation': 'query_generation',
            'execution': 'query_execution',
            'visualization': 'visualization'
          };
          
          const resultKey = `${index + 1}_${taskTypeMap[task.tool_type] || task.tool_type}`;
          const stepResult = plan.results ? plan.results[resultKey] : null;
          
          console.log(`Step ${index + 1}: ${task.task_type} -> ${resultKey}`, stepResult);
          
          return (
            <div key={`${task.task_type}_${index}`} className="plan-step">
              <div className={`step-status-icon ${stepResult?.status === 'failed' ? 'failed' : 'completed'}`}>
                {stepResult?.status === 'failed' ? '!' : '✓'}
              </div>
              <div className="step-details">
                <div className="step-name">
                  {index + 1}. {task.task_type.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
                </div>
                
                {/* Show step results */}
                {stepResult && (
                  <div className="step-output">
                    <p><strong>Result:</strong></p>
                    {stepResult.error ? (
                      <div className="step-error">
                        <p style={{color: '#dc2626'}}><strong>Error:</strong> {stepResult.error}</p>
                      </div>
                    ) : (
                      <div className="step-success">
                        {/* Show discovered tables */}
                        {stepResult.discovered_tables && (
                          <div>
                            <p><strong>Discovered Tables ({stepResult.discovered_tables.length}):</strong></p>
                            <ul>
                              {stepResult.discovered_tables.map((table: string, idx: number) => (
                                <li key={idx}>{table}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                        
                        {/* Show table suggestions */}
                        {stepResult.table_suggestions && (
                          <div>
                            <p><strong>Table Suggestions ({stepResult.table_suggestions.length}):</strong></p>
                            <small style={{color: '#6b7280', fontStyle: 'italic'}}>
                              * Empty tables (0 rows) are automatically filtered out
                            </small>
                            <ul>
                              {stepResult.table_suggestions.map((suggestion: any, idx: number) => (
                                <li key={idx}>
                                  <strong>{suggestion.rank}. {suggestion.table_name}</strong> 
                                  (Relevance: {suggestion.estimated_relevance} - {(suggestion.relevance_score * 100).toFixed(1)}%)
                                  {suggestion.row_count && suggestion.row_count !== "Unknown" && (
                                    <span style={{color: '#059669'}}> - {suggestion.row_count.toLocaleString()} records</span>
                                  )}
                                  {suggestion.row_count === "Unknown" && (
                                    <span style={{color: '#f59e0b'}}> - Row count unavailable</span>
                                  )}
                                  <br/>
                                  <small>{suggestion.description}</small>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        
                        {/* Show approved tables */}
                        {stepResult.approved_tables && (
                          <div>
                            <p><strong>Selected Tables ({stepResult.approved_tables.length}):</strong></p>
                            <ul>
                              {stepResult.approved_tables.map((table: string, idx: number) => (
                                <li key={idx} style={{color: '#059669', fontWeight: 'bold'}}>{table}</li>
                              ))}
                            </ul>
                            <p><small>Selection Method: {stepResult.selection_method} (Confidence: {stepResult.confidence})</small></p>
                          </div>
                        )}
                        
                        {/* Show semantic analysis */}
                        {stepResult.intent && (
                          <div>
                            <p><strong>Query Intent:</strong> {stepResult.intent}</p>
                            <p><strong>Complexity Score:</strong> {stepResult.complexity_score}</p>
                            {stepResult.entities && stepResult.entities.length > 0 && (
                              <p><strong>Entities Found:</strong> {stepResult.entities.join(', ')}</p>
                            )}
                          </div>
                        )}
                        
                        {/* Show similarity matching */}
                        {stepResult.matched_tables && (
                          <div>
                            <p><strong>Matched Tables ({stepResult.matched_tables.length}):</strong></p>
                            <ul>
                              {stepResult.matched_tables.map((table: string, idx: number) => (
                                <li key={idx}>{table} (Score: {stepResult.similarity_scores[idx]})</li>
                              ))}
                            </ul>
                            <p><strong>Confidence:</strong> {stepResult.confidence}</p>
                          </div>
                        )}
                        
                        {/* Show SQL query if available */}
                        {stepResult.sql && (
                          <div>
                            <p><strong>Generated SQL:</strong></p>
                            <pre className="sql-code">{stepResult.sql}</pre>
                          </div>
                        )}
                        
                        {/* Show data results if available - check multiple possible result keys */}
                        {(stepResult.data || stepResult.results) && Array.isArray(stepResult.data || stepResult.results) && (stepResult.data || stepResult.results).length > 0 && (
                          <div>
                            {(() => {
                              const dataArray = stepResult.data || stepResult.results;
                              return (
                                <>
                                  <p><strong>Query Results ({dataArray.length} rows):</strong></p>
                                  <div className="results-table" style={{
                                    maxHeight: '400px', 
                                    overflowY: 'auto', 
                                    border: '1px solid #e0e0e0', 
                                    borderRadius: '4px',
                                    marginTop: '10px'
                                  }}>
                                    <table style={{width: '100%', borderCollapse: 'collapse'}}>
                                      <thead style={{background: '#f8f9fa', position: 'sticky', top: 0}}>
                                        <tr>
                                          {Object.keys(dataArray[0]).map((key: string) => (
                                            <th key={key} style={{
                                              padding: '8px 12px', 
                                              borderBottom: '2px solid #dee2e6',
                                              fontWeight: 'bold',
                                              textAlign: 'left'
                                            }}>{key}</th>
                                          ))}
                                        </tr>
                                      </thead>
                                      <tbody>
                                        {dataArray.slice(0, 10).map((row: any, idx: number) => (
                                          <tr key={idx} style={{borderBottom: '1px solid #e9ecef'}}>
                                            {Object.values(row).map((value, vidx) => (
                                              <td key={vidx} style={{
                                                padding: '8px 12px',
                                                maxWidth: '200px',
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                                whiteSpace: 'nowrap'
                                              }}>{String(value)}</td>
                                            ))}
                                          </tr>
                                        ))}
                                      </tbody>
                                    </table>
                                    {dataArray.length > 10 && (
                                      <p style={{
                                        fontSize: '12px', 
                                        color: '#6b7280', 
                                        padding: '10px',
                                        margin: 0,
                                        background: '#f8f9fa',
                                        borderTop: '1px solid #e9ecef'
                                      }}>... and {dataArray.length - 10} more rows</p>
                                    )}
                                  </div>
                                </>
                              );
                            })()}
                          </div>
                        )}
                        
                        {/* Show visualization info */}
                        {stepResult.chart_config && (
                          <div>
                            <p><strong>Visualization Created:</strong></p>
                            <div className="chart-placeholder">
                              📊 Chart Type: {stepResult.chart_config.chart_type || 'Data Visualization'}
                            </div>
                          </div>
                        )}
                        
                        {/* Show general status */}
                        {stepResult.status && stepResult.status === 'completed' && !stepResult.error && (
                          <p style={{color: '#059669'}}>✅ Step completed successfully</p>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
            </div>
          )}
        </div>
      )}

      {/* Enhanced Results & Visualization Section */}
      {finalTask && (
        <div className="results-visualization-container">
          <div className="results-header">
            <div className="results-icon">📊</div>
            <div className="results-title-section">
              <h3 className="results-title">Results & Visualization</h3>
              <p className="results-subtitle">Your analysis results and generated charts</p>
            </div>
          </div>
          <div className="results-content">
            {(() => {
              // Fix the result key mapping for final step
              const taskTypeMap: { [key: string]: string } = {
                'schema_discovery': 'discover_schema',
                'semantic_understanding': 'semantic_analysis', 
                'similarity_matching': 'similarity_matching',
                'user_interaction': 'user_verification',
                'query_generation': 'query_generation',
                'execution': 'query_execution',
                'visualization': 'visualization'
              };
              
              const finalIndex = plan.tasks.length - 1;
              const resultKey = `${finalIndex + 1}_${taskTypeMap[finalTask.tool_type] || finalTask.tool_type}`;
              let stepResult = plan.results?.[resultKey];
              
              // NEW: Fallback - search for charts in any result if not found in expected key
              if (!stepResult?.charts) {
                console.log('Charts not found in expected key, searching all results...');
                for (const [key, result] of Object.entries(plan.results || {})) {
                  if (result && typeof result === 'object' && (result as any).charts) {
                    console.log(`Found charts in ${key}:`, (result as any).charts);
                    stepResult = result as any;
                    break;
                  }
                }
              }
              
              // Debug logging
              console.log('Final step debugging:', {
                finalIndex,
                finalTaskType: finalTask.tool_type,
                resultKey,
                stepResult,
                allResultKeys: Object.keys(plan.results || {}),
                charts: stepResult?.charts,
                chartsLength: stepResult?.charts?.length
              });
              
              // NEW: Enhanced debugging for chart detection
              console.log('=== COMPREHENSIVE CHART DEBUG ===');
              console.log('All plan results:', plan.results);
              console.log('Looking for visualization results in keys:', Object.keys(plan.results || {}));
              
              // Check all result keys for chart data
              Object.keys(plan.results || {}).forEach(key => {
                const result = plan.results[key];
                console.log(`Result ${key}:`, {
                  hasCharts: !!result.charts,
                  chartsLength: result.charts?.length,
                  chartTypes: result.charts?.map((c: any) => c.type),
                  allKeys: Object.keys(result)
                });
              });
              
              // Additional debug for charts
              if (stepResult?.charts) {
                console.log('Charts details:', stepResult.charts.map((chart: any, idx: number) => ({
                  index: idx,
                  type: chart.type,
                  title: chart.title,
                  hasData: !!chart.data,
                  dataType: typeof chart.data,
                  dataKeys: chart.data && typeof chart.data === 'object' ? Object.keys(chart.data) : 'not an object'
                })));
              }
              
              return (
                <div className="step-result">
                  {stepResult?.error ? (
                    <div className="step-error">
                      <p style={{color: '#DC2626'}}>❌ {stepResult.error}</p>
                    </div>
                  ) : stepResult ? (
                    <div>
                      {/* Show visualization */}
                      {stepResult.chart_config && (
                        <div>
                          <p><strong>Visualization Created:</strong></p>
                          <div className="chart-placeholder">
                            📊 Chart Type: {stepResult.chart_config.chart_type || 'Data Visualization'}
                          </div>
                        </div>
                      )}
                      
                      {/* Enhanced Charts Display */}
                      {stepResult.charts && Array.isArray(stepResult.charts) && stepResult.charts.length > 0 && (
                        <div className="charts-container">
                          <div className="charts-header">
                            <h4 className="charts-title">Generated Visualizations ({stepResult.charts.length})</h4>
                            <p className="charts-description">Interactive charts created from your data analysis</p>
                          </div>
                          <div className="charts-grid">
                            {stepResult.charts.map((chart: any, idx: number) => (
                              <div key={idx} className="enhanced-chart-card">
                                <div className="chart-card-header">
                                  <div className="chart-type-badge">{chart.type}</div>
                                  <h5 className="chart-title">{chart.title || 'Data Visualization'}</h5>
                                </div>
                                
                                <div className="chart-content">
                                  {/* Render matplotlib charts (base64 images) */}
                                  {chart.type === 'matplotlib' && chart.data && (
                                    <img 
                                      src={chart.data} 
                                      alt={chart.title || 'Visualization'}
                                      className="chart-image"
                                    />
                                  )}
                                  
                                  {/* Render plotly charts */}
                                  {chart.type === 'plotly' && chart.data && (
                                    <div className="plotly-chart-container">
                                      <Plot
                                        data={chart.data.data || []}
                                        layout={{
                                          ...(chart.data.layout || {}),
                                          autosize: true,
                                          margin: { t: 40, r: 20, b: 40, l: 40 },
                                          paper_bgcolor: 'rgba(0,0,0,0)',
                                          plot_bgcolor: 'rgba(0,0,0,0)'
                                        }}
                                        config={{
                                          displayModeBar: true,
                                          responsive: true,
                                          displaylogo: false,
                                          modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
                                        }}
                                        style={{width: '100%', height: '100%'}}
                                      />
                                    </div>
                                  )}
                                  
                                  {/* Fallback for other chart types */}
                                  {!['matplotlib', 'plotly'].includes(chart.type) && (
                                    <div className="chart-fallback">
                                      <div className="fallback-icon">📊</div>
                                      <p className="fallback-text">Chart data available ({chart.type})</p>
                                    </div>
                                  )}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {/* Show summary */}
                      {stepResult.summary && (
                        <div>
                          <p><strong>Summary:</strong></p>
                          <p>{stepResult.summary}</p>
                        </div>
                      )}
                      
                      {/* Show final status */}
                      {stepResult.status === 'completed' && (
                        <p style={{color: '#059669'}}>✅ Analysis completed successfully</p>
                      )}
                    </div>
                  ) : (
                    <p style={{color: '#6B7280'}}>⏳ Generating results...</p>
                  )}
                </div>
              );
            })()}
          </div>
        </div>
      )}

      {plan.estimated_cost && plan.estimated_cost > 0 && (
        <div className="plan-cost">
          <strong>Estimated cost:</strong> ${plan.estimated_cost.toFixed(3)}
        </div>
      )}
    </div>
    );
  };

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
            <button 
              className={`new-chat-btn ${isLoading ? 'loading' : ''}`}
              onClick={handleNewConversation}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <FiLoader size={12} className="spin" /> Creating...
                </>
              ) : (
                <>
                  <FiPlus size={12} /> New
                </>
              )}
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
              <div className="loading-content">
                {activePlan ? (
                  <div>
                    <span>Processing: {activePlan.current_step || 'Initializing...'}</span>
                    <div className="progress-bar" style={{
                      width: '200px',
                      height: '4px',
                      backgroundColor: '#e0e0e0',
                      borderRadius: '2px',
                      margin: '8px 0',
                      overflow: 'hidden'
                    }}>
                      <div 
                        className="progress-fill"
                        style={{
                          width: `${activePlan.progress || 0}%`,
                          height: '100%',
                          backgroundColor: '#4285f4',
                          borderRadius: '2px',
                          transition: 'width 0.3s ease'
                        }}
                      />
                    </div>
                    <span style={{fontSize: '12px', color: '#666'}}>
                      {activePlan.progress || 0}% complete - This may take 1-2 minutes
                    </span>
                  </div>
                ) : (
                  <span>AI is analyzing your request...</span>
                )}
              </div>
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
