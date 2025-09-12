import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import './EnhancedPharmaChat.css';
import { 
  FiSend, FiCopy, FiCheck, FiDatabase, FiSettings, 
  FiBarChart2, FiTable, FiAlertTriangle, FiInfo, 
  FiChevronDown, FiChevronUp, FiLoader, FiUser,
  FiMessageSquare, FiClock, FiArchive, FiSearch,
  FiPlus, FiMoreHorizontal, FiDownload, FiShare2, FiZap
} from 'react-icons/fi';
import Plot from 'react-plotly.js';
import EnhancedTable from './EnhancedTable';
import ChartCustomizer from './ChartCustomizer';
import ProgressIndicator from './ProgressIndicator';
import IncrementalResults from './IncrementalResults';
import { downloadChartAsPNG, downloadChartAsPDF, downloadChartAsSVG, applyChartColors, convertChartType } from '../utils/chartUtils';

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

interface ProgressStep {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  startTime?: number;
  endTime?: number;
  progress?: number;
}

interface PartialResult {
  id: string;
  stepId: string;
  stepName: string;
  type: 'query' | 'chart' | 'insight' | 'error';
  data?: any;
  timestamp: number;
  isComplete: boolean;
  preview?: boolean;
}

interface Conversation {
  conversation_id: string;
  title: string;
  created_at: string;
  last_activity: string;
  therapeutic_area?: string;
  total_cost: number;
  total_tokens: number;
  last_activity_timestamp: number;
  is_archived: boolean;
  messages: ChatMessage[];
}

// Backend task structure (simplified from DynamicAgentOrchestrator)
interface BackendTask {
  task_type: string;  // e.g., "schema_discovery", "query_generation", "execution"
  agent: string;      // e.g., "dynamic"
}

// Enhanced frontend step structure for UI display
interface PlanStep {
  step_id: string;
  task_type: string;
  name: string;
  description: string;
  status: 'pending' | 'executing' | 'completed' | 'error';
  start_time?: string;
  end_time?: string;
  error?: string;
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
  user_query: string;
  reasoning_steps: string[];
  estimated_execution_time: string;  // e.g., "6s"
  tasks: BackendTask[];              // Backend simplified task list
  status: 'completed' | 'failed';   // Backend only returns these two
  results: { [key: string]: any };  // Task results keyed by task_id
  error?: string;                    // Present if status is "failed"
  
  // Enhanced frontend fields (derived from backend data)
  enhanced_steps?: PlanStep[];       // Enhanced step display for UI
  progress?: number;                 // Calculated progress (0-1)
  current_step?: string;             // Current step ID for tracking
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
        session_id: conversationId,
        use_deterministic: true  // Enable deterministic mode by default for testing
      })
      // Explicitly no timeout specified - browser default is usually 5+ minutes
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  },

  // Intent Detection - Check if message needs planning or casual response
  detectIntent: async (query: string, context?: any): Promise<{needsPlanning: boolean, isContextQuestion?: boolean, response?: string, contextType?: string}> => {
    const response = await fetch('http://localhost:8000/api/agent/detect-intent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, context })
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
      last_activity_timestamp: Date.now() - 60000, // 1 minute ago
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
      last_activity_timestamp: Date.now() - 3600000, // 1 hour ago
      is_archived: false,
      messages: []
    }
  ]
};

interface EnhancedPharmaChatProps {
  onNavigateToSettings?: () => void;
}

// Helper function to transform backend plan to enhanced frontend format
const enhancePlanForUI = (backendPlan: QueryPlan): QueryPlan => {
  // Task type mapping for step names and descriptions
  const taskMetadata: { [key: string]: { name: string; description: string; stepId: string } } = {
    'schema_discovery': { name: 'Schema Discovery', description: 'Discovering relevant database tables and columns', stepId: '1_discover_schema' },
    'semantic_understanding': { name: 'Semantic Analysis', description: 'Understanding query intent and extracting key entities', stepId: '2_semantic_analysis' },
    'similarity_matching': { name: 'Similarity Matching', description: 'Matching query terms to database schema', stepId: '3_similarity_matching' },
    'user_interaction': { name: 'Table Selection', description: 'Selecting the best tables for analysis', stepId: '4_user_verification' },
    'query_generation': { name: 'SQL Generation', description: 'Creating optimized database query', stepId: '5_query_generation' },
    'python_generation': { name: 'Python Generation', description: 'Generating Python code for data analysis', stepId: '5_5_python_generation' },
    'execution': { name: 'Data Retrieval', description: 'Executing query and fetching results', stepId: '6_query_execution' },
    'visualization': { name: 'Chart Creation', description: 'Generating visualizations from data', stepId: '7_visualization' }
  };

  // Create enhanced steps from backend tasks using actual backend task IDs
  const enhancedSteps: PlanStep[] = backendPlan.tasks.map((task, index) => {
    const metadata = taskMetadata[task.task_type] || { 
      name: task.task_type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()), 
      description: `Processing ${task.task_type}`,
      stepId: `${index + 1}_${task.task_type}`  // Use actual backend task ID format
    };
    
    // Use the actual backend task ID format (e.g., "1_semantic_understanding")
    const actualStepId = `${index + 1}_${task.task_type}`;
    
    // Check if this step has results (completed) using the actual step ID
    const hasResult = backendPlan.results && backendPlan.results[actualStepId];
    const stepStatus = backendPlan.status === 'failed' ? 'error' : 
                     hasResult ? 'completed' : 'pending';
    
    return {
      step_id: actualStepId,  // Use actual backend task ID
      task_type: task.task_type,
      name: metadata.name,
      description: metadata.description,
      status: stepStatus,
      output_data: hasResult ? backendPlan.results[actualStepId] : undefined
    };
  });

  // Calculate progress and current step
  const completedSteps = enhancedSteps.filter(step => step.status === 'completed').length;
  const totalSteps = enhancedSteps.length;
  const progress = totalSteps > 0 ? completedSteps / totalSteps : 0;
  
  // Find current step (first non-completed step, or null if all done)
  const currentStep = enhancedSteps.find(step => step.status !== 'completed')?.step_id || null;

  return {
    ...backendPlan,
    enhanced_steps: enhancedSteps,
    progress: progress,
    current_step: currentStep || undefined
  };
};

const EnhancedPharmaChat: React.FC<EnhancedPharmaChatProps> = ({ onNavigateToSettings }) => {
  // State Management
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversation, setActiveConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [messageResults, setMessageResults] = useState<{[messageId: string]: any}>({});
  const [currentMessage, setCurrentMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activePlan, setActivePlan] = useState<QueryPlan | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showProfile, setShowProfile] = useState(false);
  const [showQuickActions, setShowQuickActions] = useState(false);
  const [expandedSteps, setExpandedSteps] = useState<{[key: string]: boolean}>({});
  const [showStepsDetails, setShowStepsDetails] = useState(false);
  const [selectedStepKey, setSelectedStepKey] = useState<string | null>(null);
  const [currentContext, setCurrentContext] = useState<any>(null); // Track current analysis context
  const [databaseStatus, setDatabaseStatus] = useState<DatabaseStatus>({
    isConnected: false,
    databaseType: 'Unknown',
    server: '',
    database: '',
    schema: '',
    warehouse: ''
  });

  // Chart customization state
  const [chartCustomizations, setChartCustomizations] = useState<{[key: string]: any}>({});
  const [customChartTypes, setCustomChartTypes] = useState<{[key: string]: string}>({});
  const [customColors, setCustomColors] = useState<{[key: string]: string[]}>({});
  
  // Code display state
  const [expandedCodeStep, setExpandedCodeStep] = useState<string | null>(null);

  // Progress tracking state
  const [progressSteps, setProgressSteps] = useState<ProgressStep[]>([]);
  const [currentProgressStep, setCurrentProgressStep] = useState<string | null>(null);
  const [analysisStartTime, setAnalysisStartTime] = useState<number | null>(null);
  const [estimatedTotalTime, setEstimatedTotalTime] = useState<number | null>(null);
  const [showProgress, setShowProgress] = useState(false);

  // Incremental results state
  const [incrementalResults, setIncrementalResults] = useState<PartialResult[]>([]);
  const [showIncrementalResults, setShowIncrementalResults] = useState(true);

  // WebSocket for real-time progress
  const [websocket, setWebsocket] = useState<WebSocket | null>(null);

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Load persisted state from localStorage on component mount
  useEffect(() => {
    try {
      const savedMessages = localStorage.getItem('nl2q-chat-messages');
      const savedMessageResults = localStorage.getItem('nl2q-message-results');
      const savedConversations = localStorage.getItem('nl2q-conversations');
      const savedActiveConversation = localStorage.getItem('nl2q-active-conversation');

      if (savedMessages) {
        setMessages(JSON.parse(savedMessages));
      }
      if (savedMessageResults) {
        setMessageResults(JSON.parse(savedMessageResults));
      }
      if (savedConversations) {
        setConversations(JSON.parse(savedConversations));
      }
      if (savedActiveConversation) {
        setActiveConversation(JSON.parse(savedActiveConversation));
      }
    } catch (error) {
      console.error('Error loading persisted chat state:', error);
    }
  }, []);

  // Persist messages to localStorage whenever they change (conversation-specific)
  useEffect(() => {
    if (messages.length > 0 && activeConversation) {
      // Save to conversation-specific storage
      localStorage.setItem(`nl2q-conversation-${activeConversation.conversation_id}`, JSON.stringify({
        messages: messages,
        messageResults: messageResults,
        activePlan: activePlan,
        selectedStepKey: selectedStepKey,
        showStepsDetails: showStepsDetails
      }));
      
      // Also save to legacy storage for backwards compatibility
      localStorage.setItem('nl2q-chat-messages', JSON.stringify(messages));
    }
  }, [messages, messageResults, activePlan, selectedStepKey, showStepsDetails, activeConversation]);

  // Persist message results to localStorage whenever they change
  useEffect(() => {
    if (Object.keys(messageResults).length > 0) {
      localStorage.setItem('nl2q-message-results', JSON.stringify(messageResults));
    }
  }, [messageResults]);

  // Persist conversations to localStorage whenever they change
  useEffect(() => {
    if (conversations.length > 0) {
      localStorage.setItem('nl2q-conversations', JSON.stringify(conversations));
    }
  }, [conversations]);

  // Persist active conversation to localStorage whenever it changes
  useEffect(() => {
    if (activeConversation) {
      localStorage.setItem('nl2q-active-conversation', JSON.stringify(activeConversation));
    }
  }, [activeConversation]);
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

        // Only load conversations if not already persisted
        const hasPersistedConversations = localStorage.getItem('nl2q-conversations');
        if (!hasPersistedConversations) {
          // Load conversations (using mock for now)
          const convs = api.mockGetConversations();
          setConversations(convs);

          // Set first conversation as active
          if (convs.length > 0) {
            setActiveConversation(convs[0]);
            // In real implementation, load messages here
          }
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

  // Setup WebSocket connection for real-time progress
  useEffect(() => {
    const connectWebSocket = () => {
      const ws = new WebSocket('ws://localhost:8000/ws/progress');
      
      ws.onopen = () => {
        console.log('Connected to progress WebSocket');
        setWebsocket(ws);
      };
      
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          console.log('Received WebSocket message:', message);
          
          // Handle different message types
          if (message.type === 'execution_progress') {
            // New execution progress format
            const progressData = message.data;
            
            if (progressData.stage === 'execution_started') {
              // Initialize progress steps from backend
              const steps = progressData.tasks?.map((task: any, index: number) => ({
                id: task.id,
                name: task.type.replace('_', ' ').split(' ').map((word: string) => 
                  word.charAt(0).toUpperCase() + word.slice(1)
                ).join(' '),
                status: 'pending' as const,
                progress: 0
              })) || [];
              
              setProgressSteps(steps);
              setShowProgress(true);
              setAnalysisStartTime(Date.now());
              setEstimatedTotalTime(30); // Default estimate
              
            } else if (progressData.stage === 'task_started') {
              setCurrentProgressStep(progressData.currentStep);
              updateProgressStep(progressData.currentStep, 'running', progressData.progress);
              
            } else if (progressData.stage === 'task_completed') {
              updateProgressStep(progressData.currentStep, 'completed', 100);
              
            } else if (progressData.stage === 'task_error') {
              updateProgressStep(progressData.currentStep, 'error', progressData.progress);
              
            } else if (progressData.stage === 'execution_completed') {
              // All tasks completed - hide progress after a delay
              setCurrentProgressStep(null);
              setTimeout(() => {
                setShowProgress(false);
                setProgressSteps([]);
              }, 2000); // Hide after 2 seconds to show completion
            }
            
          } else if (message.type === 'indexing_progress') {
            // Handle indexing progress from schema indexing/force ingestion
            const indexingData = message.data;
            console.log('📊 Indexing progress:', indexingData);
            
            // You can add specific UI updates for indexing progress here
            if (indexingData.isIndexing) {
              console.log(`Indexing: ${indexingData.stage} - ${indexingData.processedTables}/${indexingData.totalTables} tables`);
            }
            
          } else {
            // Legacy format - treat as execution progress
            const progressData = message;
            
            if (progressData.stage === 'execution_started') {
              // Initialize progress steps from backend
              const steps = progressData.tasks?.map((task: any, index: number) => ({
                id: task.id,
                name: task.type.replace('_', ' ').split(' ').map((word: string) => 
                  word.charAt(0).toUpperCase() + word.slice(1)
                ).join(' '),
                status: 'pending' as const,
                progress: 0
              })) || [];
              
              setProgressSteps(steps);
              setShowProgress(true);
              setAnalysisStartTime(Date.now());
              setEstimatedTotalTime(30); // Default estimate
              
            } else if (progressData.stage === 'task_started') {
              setCurrentProgressStep(progressData.currentStep);
              updateProgressStep(progressData.currentStep, 'running', progressData.progress);
              
            } else if (progressData.stage === 'task_completed') {
              updateProgressStep(progressData.currentStep, 'completed', 100);
              
            } else if (progressData.stage === 'task_error') {
              updateProgressStep(progressData.currentStep, 'error', progressData.progress);
            }
          }
          
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      ws.onclose = () => {
        console.log('WebSocket connection closed, attempting to reconnect...');
        setTimeout(connectWebSocket, 3000); // Reconnect after 3 seconds
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    };
    
    connectWebSocket();
    
    // Cleanup WebSocket when component unmounts
    return () => {
      if (websocket) {
        websocket.close();
      }
    };
  }, []);

  // Auto-cleanup WebSocket
  useEffect(() => {
    return () => {
      if (websocket) {
        websocket.close();
      }
    };
  }, [websocket]);

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

  // Auto-hide progress indicator after completion
  useEffect(() => {
    const completedSteps = progressSteps.filter(step => step.status === 'completed').length;
    const totalSteps = progressSteps.length;
    
    if (totalSteps > 0 && completedSteps === totalSteps) {
      // All steps completed - hide progress after 3 seconds
      const timer = setTimeout(() => {
        setShowProgress(false);
        setProgressSteps([]);
        setCurrentProgressStep(null);
      }, 3000);
      
      return () => clearTimeout(timer);
    } else if (totalSteps > 0 && completedSteps < totalSteps) {
      // Progress is active - show it
      setShowProgress(true);
    }
  }, [progressSteps]);

  // Progress tracking functions
  const initializeProgress = (steps: string[], estimatedTime?: number) => {
    const progressSteps = steps.map((stepName, index) => ({
      id: `step_${index}`,
      name: stepName,
      status: 'pending' as const
    }));
    
    setProgressSteps(progressSteps);
    setCurrentProgressStep(null);
    setAnalysisStartTime(Date.now());
    setEstimatedTotalTime(estimatedTime || null);
    setIncrementalResults([]);
  };

  const updateProgressStep = (stepId: string, status: 'running' | 'completed' | 'error', progress?: number) => {
    setProgressSteps(prev => prev.map(step => {
      if (step.id === stepId) {
        const updatedStep = { 
          ...step, 
          status, 
          progress: progress || step.progress
        };
        
        if (status === 'running' && !step.startTime) {
          updatedStep.startTime = Date.now();
        } else if ((status === 'completed' || status === 'error') && step.startTime) {
          updatedStep.endTime = Date.now();
        }
        
        return updatedStep;
      }
      return step;
    }));

    if (status === 'running') {
      setCurrentProgressStep(stepId);
    } else if (status === 'completed' || status === 'error') {
      // Move to next step if current step is completed
      const currentIndex = progressSteps.findIndex(s => s.id === stepId);
      const nextStep = progressSteps[currentIndex + 1];
      if (nextStep && nextStep.status === 'pending') {
        setCurrentProgressStep(nextStep.id);
      } else {
        setCurrentProgressStep(null);
      }
    }
  };

  const addIncrementalResult = (result: Omit<PartialResult, 'timestamp' | 'id'>) => {
    const newResult: PartialResult = {
      ...result,
      id: `result_${Date.now()}_${Math.random()}`,
      timestamp: Date.now()
    };
    
    setIncrementalResults(prev => [...prev, newResult]);
    return newResult.id;
  };

  const updateIncrementalResult = (resultId: string, updates: Partial<PartialResult>) => {
    setIncrementalResults(prev => prev.map(result => 
      result.id === resultId ? { ...result, ...updates } : result
    ));
  };

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
    
    // Update conversation activity timestamp
    if (activeConversation) {
      const updatedConversation = {
        ...activeConversation,
        last_activity_timestamp: Date.now(),
        last_activity: new Date().toISOString()
      };
      setActiveConversation(updatedConversation);
      setConversations(prev => 
        prev.map(conv => 
          conv.conversation_id === activeConversation.conversation_id 
            ? updatedConversation 
            : conv
        )
      );
    }
    
    const messageContent = currentMessage;
    setCurrentMessage('');
    setIsLoading(true);
    setAnalysisStartTime(Date.now()); // Start timing the analysis

    try {
      // Build context from current analysis for intent detection
      let analysisContext = currentContext || {};
      if (activePlan && activePlan.status === 'completed') {
        // Extract context from completed plan
        const planContext: any = {
          hasCharts: false,
          hasTable: false,
          chartTypes: [],
          keyInsights: [],
          lastAnalysis: '',
          tableData: null,
          chartData: null,
          tableColumns: []
        };

        // Check for charts and data in results
        Object.values(activePlan.results || {}).forEach((result: any) => {
          if (result.charts && result.charts.length > 0) {
            planContext.hasCharts = true;
            result.charts.forEach((chart: any) => {
              if (chart.type && !planContext.chartTypes.includes(chart.type)) {
                planContext.chartTypes.push(chart.type);
              }
            });
            
            // Extract clean chart data (just the values, not styling)
            const cleanChartData = result.charts.map((chart: any) => {
              if (chart.data && chart.data.data && chart.data.data[0]) {
                const plotData = chart.data.data[0];
                return {
                  type: chart.type,
                  title: chart.data.layout?.title?.text || 'Chart',
                  data_points: {
                    x_values: plotData.x || [],
                    y_values: plotData.y || [],
                    labels: plotData.x || [],
                    values: plotData.y || []
                  }
                };
              }
              return { type: chart.type, data_points: {} };
            });
            
            planContext.chartData = cleanChartData;
          }
          
          if (result.data || result.table_data) {
            planContext.hasTable = true;
            const tableData = result.data || result.table_data;
            
            // Include actual table data for context (limit to first few rows)
            if (Array.isArray(tableData) && tableData.length > 0) {
              planContext.tableData = tableData.slice(0, 10); // First 10 rows
              planContext.tableColumns = Object.keys(tableData[0] || {});
              planContext.rowCount = tableData.length;
            }
          }
          
          if (result.summary) {
            planContext.lastAnalysis += result.summary + ' ';
          }
        });

        // Add plan-level insights
        if (activePlan.reasoning_steps) {
          planContext.keyInsights = activePlan.reasoning_steps.slice(0, 3);
        }

        analysisContext = { ...analysisContext, ...planContext };
      }

      // First, detect intent to see if this needs planning or just casual response
      console.log('Detecting intent for message:', messageContent);
      console.log('Current analysis context:', analysisContext);
      
      let intentResult;
      try {
        intentResult = await api.detectIntent(messageContent, analysisContext);
        console.log('Intent detection result:', intentResult);
      } catch (intentError) {
        console.warn('Intent detection failed, falling back to planning:', intentError);
        // If intent detection fails, default to planning (backward compatibility)
        intentResult = { needsPlanning: true };
      }

      if (!intentResult.needsPlanning) {
        // Handle as casual conversation or context question
        console.log(intentResult.isContextQuestion ? 'Handling context question' : 'Treating as casual conversation');
        setIsLoading(false);
        
        // Clear activePlan for conversational responses to prevent showing old execution plans
        setActivePlan(null);
        
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
      
      // Progress will be initialized via WebSocket when backend starts execution
      // Reset any existing progress
      setProgressSteps([]);
      setCurrentProgressStep(null);
      setIncrementalResults([]);
      
      const planResult = await api.sendQuery(messageContent, userProfile.user_id, activeConversation.conversation_id);
      
      // Add debugging to see what we're getting
      console.log('Received plan result:', planResult);
      console.log('Plan status:', planResult.status);
      console.log('Plan tasks:', planResult.tasks);
      
      // Check if this plan contains meaningful database operations
      const hasDbOperations = planResult.tasks && planResult.tasks.some((task: any) => 
        task.task_type && ['schema_discovery', 'query_generation', 'execution', 'sql_generation'].includes(task.task_type)
      );
      
      // Only set activePlan if there are actual database operations
      if (hasDbOperations) {
        // Enhance the plan for UI display
        const enhancedPlan = enhancePlanForUI(planResult);
        setActivePlan(enhancedPlan);
      } else {
        // For simple conversational responses, clear the plan
        setActivePlan(null);
      }
      
      // Progress updates will be handled via WebSocket real-time updates
      // Add incremental results for completed results
      if (planResult.status === 'completed' && planResult.results) {
        Object.entries(planResult.results).forEach(([stepId, result]: [string, any]) => {
          if (result.results && Array.isArray(result.results) && result.results.length > 0) {
            addIncrementalResult({
              stepId,
              stepName: `Data Query Results`,
              type: 'query',
              data: result.results,
              isComplete: true
            });
          }
          
          if (result.charts && Array.isArray(result.charts) && result.charts.length > 0) {
            result.charts.forEach((chart: any, chartIndex: number) => {
              addIncrementalResult({
                stepId: `${stepId}_chart_${chartIndex}`,
                stepName: `${chart.title || 'Visualization'}`,
                type: 'chart',
                data: chart.data,
                isComplete: true
              });
            });
          }
        });
      }
      
      // If plan is completed, add assistant response to messages and clear active plan
      if (planResult.status === 'completed' || planResult.status === 'failed') {
        setIsLoading(false);
        // Clear progress tracking since analysis is complete
        setAnalysisStartTime(null);
        
        // Update context with new analysis results if completed
        if (planResult.status === 'completed') {
          const newContext: any = {
            hasCharts: false,
            hasTable: false,
            chartTypes: [],
            keyInsights: [],
            lastAnalysis: ''
          };

          // Extract context from results
          Object.values(planResult.results || {}).forEach((result: any) => {
            if (result.charts && result.charts.length > 0) {
              newContext.hasCharts = true;
              result.charts.forEach((chart: any) => {
                if (chart.type && !newContext.chartTypes.includes(chart.type)) {
                  newContext.chartTypes.push(chart.type);
                }
              });
              
              // Preserve actual chart data for follow-ups
              const cleanChartData = result.charts.map((chart: any) => {
                if (chart.data && chart.data.data && chart.data.data[0]) {
                  const plotData = chart.data.data[0];
                  return {
                    type: chart.type,
                    title: chart.data.layout?.title?.text || 'Chart',
                    data_points: {
                      x_values: plotData.x || [],
                      y_values: plotData.y || [],
                      labels: plotData.x || [],
                      values: plotData.y || []
                    }
                  };
                }
                return { type: chart.type, data_points: {} };
              });
              newContext.chartData = cleanChartData;
            }
            
            if (result.data || result.table_data) {
              newContext.hasTable = true;
              const tableData = result.data || result.table_data;
              
              // Preserve actual table data for follow-ups
              if (Array.isArray(tableData) && tableData.length > 0) {
                newContext.tableData = tableData.slice(0, 50); // Keep more rows for follow-ups
                newContext.tableColumns = Object.keys(tableData[0] || {});
                newContext.rowCount = tableData.length;
              }
            }
            
            if (result.summary) {
              newContext.lastAnalysis += result.summary + ' ';
            }
          });

          // Add reasoning insights
          if (planResult.reasoning_steps) {
            newContext.keyInsights = planResult.reasoning_steps.slice(0, 3);
          }

          setCurrentContext(newContext);
          console.log('Updated context with new analysis:', newContext);
        }
        
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
        
        // Store results with this message
        if (planResult.status === 'completed' && planResult.results) {
          console.log('Storing results for message:', assistantMessage.message_id, {
            planResult: planResult,
            results: planResult.results
          });
          setMessageResults(prev => ({
            ...prev,
            [assistantMessage.message_id]: {
              plan: planResult,
              results: planResult.results
            }
          }));
        }
        
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
              
              // Store results with this message
              if (updatedPlan.status === 'completed' && updatedPlan.results) {
                console.log('Storing results for polled message:', assistantMessage.message_id, {
                  updatedPlan: updatedPlan,
                  results: updatedPlan.results
                });
                setMessageResults(prev => ({
                  ...prev,
                  [assistantMessage.message_id]: {
                    plan: updatedPlan,
                    results: updatedPlan.results
                  }
                }));
              }
              
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
      
      // Save current conversation state before creating new one
      if (activeConversation) {
        setConversations(prev => prev.map(conv => 
          conv.conversation_id === activeConversation.conversation_id 
            ? { ...conv, messages: messages }
            : conv
        ));
        
        localStorage.setItem(`nl2q-conversation-${activeConversation.conversation_id}`, JSON.stringify({
          messages: messages,
          messageResults: messageResults,
          activePlan: activePlan,
          selectedStepKey: selectedStepKey,
          showStepsDetails: showStepsDetails
        }));
      }
      
      const newConv = await api.createConversation(userProfile.user_id);
      
      // Validate the response
      if (!newConv || !newConv.conversation_id) {
        throw new Error('Invalid conversation response from server');
      }
      
      setConversations(prev => [newConv, ...prev]);
      setActiveConversation(newConv);
      setMessages([]);
      setMessageResults({});
      setActivePlan(null);
      setSelectedStepKey(null);
      setShowStepsDetails(false);
      setProgressSteps([]);
      setCurrentProgressStep(null);
      setIncrementalResults([]);
      setShowProgress(false);
      
      // Clear localStorage for fresh start
      localStorage.removeItem('nl2q-chat-messages');
      localStorage.removeItem('nl2q-message-results');
      
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
        last_activity_timestamp: Date.now(),
        is_archived: false,
        messages: []
      };
      
      setConversations(prev => [fallbackConv, ...prev]);
      setActiveConversation(fallbackConv);
      setMessages([]);
      setMessageResults({});
      setActivePlan(null);
      setSelectedStepKey(null);
      setShowStepsDetails(false);
      
      // Clear localStorage for fresh start
      localStorage.removeItem('nl2q-chat-messages');
      localStorage.removeItem('nl2q-message-results');
      
      console.log('Created fallback conversation:', fallbackConv.conversation_id);
    } finally {
      setIsLoading(false);
    }
  };

  // Switch conversation
  const handleConversationSelect = (conversation: Conversation) => {
    // Save current conversation state before switching
    if (activeConversation) {
      // Update the current conversation's messages in the conversations list
      setConversations(prev => prev.map(conv => 
        conv.conversation_id === activeConversation.conversation_id 
          ? { ...conv, messages: messages }
          : conv
      ));
      
      // Save current state to localStorage for this conversation
      localStorage.setItem(`nl2q-conversation-${activeConversation.conversation_id}`, JSON.stringify({
        messages: messages,
        messageResults: messageResults,
        activePlan: activePlan,
        selectedStepKey: selectedStepKey,
        showStepsDetails: showStepsDetails
      }));
    }

    // Load the selected conversation state
    setActiveConversation(conversation);
    
    // Try to load saved state for this conversation
    const savedState = localStorage.getItem(`nl2q-conversation-${conversation.conversation_id}`);
    if (savedState) {
      try {
        const parsedState = JSON.parse(savedState);
        setMessages(parsedState.messages || conversation.messages || []);
        setMessageResults(parsedState.messageResults || {});
        setActivePlan(parsedState.activePlan || null);
        setSelectedStepKey(parsedState.selectedStepKey || null);
        setShowStepsDetails(parsedState.showStepsDetails || false);
      } catch (error) {
        console.error('Error loading conversation state:', error);
        // Fallback to conversation messages
        const sortedMessages = (conversation.messages || []).sort((a, b) => 
          new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
        );
        setMessages(sortedMessages);
        setMessageResults({});
        setActivePlan(null);
        setSelectedStepKey(null);
        setShowStepsDetails(false);
      }
    } else {
      // Sort messages by timestamp to ensure correct order
      const sortedMessages = (conversation.messages || []).sort((a, b) => 
        new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      );
      setMessages(sortedMessages);
      setMessageResults({});
      setActivePlan(null);
      setSelectedStepKey(null);
      setShowStepsDetails(false);
    }
    
    // Clear any active progress/results since we're switching conversations
    setProgressSteps([]);
    setCurrentProgressStep(null);
    setIncrementalResults([]);
    setShowProgress(false);
  };

  // Clear current chat
  const handleClearChat = () => {
    if (!activeConversation) return;
    
    if (window.confirm('Are you sure you want to clear this chat? This action cannot be undone.')) {
      // Clear the current conversation's messages
      setMessages([]);
      setMessageResults({});
      setActivePlan(null);
      setSelectedStepKey(null);
      setShowStepsDetails(false);
      setProgressSteps([]);
      setCurrentProgressStep(null);
      setIncrementalResults([]);
      setShowProgress(false);
      
      // Update the conversation in the list
      setConversations(prev => prev.map(conv => 
        conv.conversation_id === activeConversation.conversation_id 
          ? { ...conv, messages: [] }
          : conv
      ));
      
      // Update the active conversation
      setActiveConversation(prev => prev ? { ...prev, messages: [] } : null);
      
      // Clear localStorage for this conversation
      localStorage.removeItem(`nl2q-conversation-${activeConversation.conversation_id}`);
      localStorage.removeItem('nl2q-chat-messages');
      localStorage.removeItem('nl2q-message-results');
      
      console.log('Chat cleared for conversation:', activeConversation.conversation_id);
    }
  };

  // Quick action buttons
  const quickActions = [
    "Analyze healthcare provider engagement metrics",
    "Show omnichannel marketing campaign performance", 
    "Review medical claims processing trends",
    "Track sales rep call activity and outcomes"
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

  // Format recent activity indicator
  const formatRecentActivity = (timestamp: number) => {
    const now = Date.now();
    const diffMs = now - timestamp;
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 5) return { text: 'Active now', color: '#00d084' };
    if (diffMins < 30) return { text: 'Recent', color: '#ffa500' };
    if (diffMins < 1440) return { text: 'Today', color: '#6c757d' };
    return { text: '', color: 'transparent' };
  };

  // Render plan execution with live status
  const renderPlanExecution = (plan: QueryPlan) => {
    console.log('Plan tasks:', plan.tasks);
    console.log('Plan results:', plan.results);
    console.log('Plan current step:', plan.current_step);
    console.log('Plan progress:', plan.progress);
    
    // Use enhanced steps for UI display (dynamic steps from backend)
    const enhancedSteps = plan.enhanced_steps || [];
    
    // Convert enhanced steps to the format expected by the UI
    const dynamicSteps = enhancedSteps.map((step, index) => ({
      key: step.step_id,
      name: step.name,
      description: step.description
    }));
    
    // Calculate current step index using actual dynamic steps
    const currentStepIndex = plan.status === 'completed' ? dynamicSteps.length : 
      (plan.current_step ? dynamicSteps.findIndex(step => step.key === plan.current_step) : -1);
    
    // Handle status properly for backend data
    const planStatus = plan.status;
    const isCompleted = planStatus === 'completed';
    const isFailed = planStatus === 'failed';
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
            </span>
            <FiChevronDown className={`toggle-icon ${showStepsDetails ? 'rotated' : ''}`} />
          </div>

          {/* Horizontal Steps Timeline - Dynamic Display */}
          <div className="horizontal-steps-timeline">
            {dynamicSteps.map((step, index) => {
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
                  {index < dynamicSteps.length - 1 && currentStepIndex > index && (
                    <div className={`step-connector ${isCompleted ? 'completed' : ''}`}></div>
                  )}
                </div>
              );
            })}
            
            {/* Show remaining steps count - only for executing plans */}
            {plan.status !== 'completed' && currentStepIndex < dynamicSteps.length - 1 && (
              <div className="remaining-steps-indicator">
                <div className="remaining-steps-circle">
                  <span className="remaining-count">+{dynamicSteps.length - currentStepIndex - 1}</span>
                </div>
                <div className="remaining-steps-label">More steps</div>
              </div>
            )}
          </div>

          {/* Step Details Panel - Only for visible steps */}
          {showStepsDetails && selectedStepKey && (
            <div className="step-details-panel">
              {(() => {
                const selectedStep = dynamicSteps.find(s => s.key === selectedStepKey);
                const stepIndex = dynamicSteps.findIndex(s => s.key === selectedStepKey);
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
                    {isCompleted && selectedStepKey && plan.results?.[selectedStepKey] && (
                      <div className="step-result-details">
                        {(() => {
                          const stepResult = plan.results[selectedStepKey];
                          const stepType = selectedStep?.name || 'Step';
                          
                          // Dynamic step result display based on step type
                          if (selectedStep?.name.includes('Schema') || selectedStep?.name.includes('Discovery')) {
                            return (
                              <>
                                <p>✅ Database schema discovered successfully</p>
                                <p>📊 Tables and columns identified for analysis</p>
                              </>
                            );
                          } else if (selectedStep?.name.includes('Semantic') || selectedStep?.name.includes('Understanding')) {
                            return (
                              <>
                                <p>🧠 Query intent understood</p>
                                <p>🔍 Key entities extracted and analyzed</p>
                              </>
                            );
                          } else if (selectedStep?.name.includes('Matching') || selectedStep?.name.includes('Similarity')) {
                            return (
                              <>
                                <p>🎯 Matching tables identified successfully</p>
                                <p>📈 Relevance scores calculated</p>
                              </>
                            );
                          } else if (selectedStep?.name.includes('SQL') || selectedStep?.name.includes('Generation')) {
                            return (
                              <>
                                <p>✅ SQL query generated successfully</p>
                                {stepResult?.sql_query && (
                                  <div className="sql-preview">
                                    <code>{stepResult.sql_query.substring(0, 100)}...</code>
                                    <button 
                                      className="code-expand-btn"
                                      onClick={() => setExpandedCodeStep(
                                        expandedCodeStep === selectedStepKey + '_sql' ? null : selectedStepKey + '_sql'
                                      )}
                                      style={{
                                        marginLeft: '8px',
                                        padding: '2px 8px',
                                        fontSize: '11px',
                                        background: '#3B82F6',
                                        color: 'white',
                                        border: 'none',
                                        borderRadius: '4px',
                                        cursor: 'pointer'
                                      }}
                                    >
                                      {expandedCodeStep === selectedStepKey + '_sql' ? 'Hide' : 'View Full SQL'}
                                    </button>
                                  </div>
                                )}
                                {expandedCodeStep === selectedStepKey + '_sql' && stepResult?.sql_query && (
                                  <div className="code-full-display" style={{
                                    marginTop: '8px',
                                    padding: '12px',
                                    background: '#F8F9FA',
                                    border: '1px solid #E5E7EB',
                                    borderRadius: '4px',
                                    fontFamily: 'monospace',
                                    fontSize: '12px',
                                    whiteSpace: 'pre-wrap',
                                    maxHeight: '200px',
                                    overflow: 'auto'
                                  }}>
                                    {stepResult.sql_query}
                                  </div>
                                )}
                              </>
                            );
                          } else if (selectedStep?.name.includes('Python')) {
                            return (
                              <>
                                <p>✅ Python code generated successfully</p>
                                {stepResult?.python_code && (
                                  <div className="python-preview">
                                    <code>{stepResult.python_code.substring(0, 100)}...</code>
                                    <button 
                                      className="code-expand-btn"
                                      onClick={() => setExpandedCodeStep(
                                        expandedCodeStep === selectedStepKey + '_python' ? null : selectedStepKey + '_python'
                                      )}
                                      style={{
                                        marginLeft: '8px',
                                        padding: '2px 8px',
                                        fontSize: '11px',
                                        background: '#10B981',
                                        color: 'white',
                                        border: 'none',
                                        borderRadius: '4px',
                                        cursor: 'pointer'
                                      }}
                                    >
                                      {expandedCodeStep === selectedStepKey + '_python' ? 'Hide' : 'View Full Python'}
                                    </button>
                                  </div>
                                )}
                                {expandedCodeStep === selectedStepKey + '_python' && stepResult?.python_code && (
                                  <div className="code-full-display" style={{
                                    marginTop: '8px',
                                    padding: '12px',
                                    background: '#F8F9FA',
                                    border: '1px solid #E5E7EB',
                                    borderRadius: '4px',
                                    fontFamily: 'monospace',
                                    fontSize: '12px',
                                    whiteSpace: 'pre-wrap',
                                    maxHeight: '200px',
                                    overflow: 'auto'
                                  }}>
                                    {stepResult.python_code}
                                  </div>
                                )}
                              </>
                            );
                          } else if (selectedStep?.name.includes('Execution') || selectedStep?.name.includes('Retrieval')) {
                            return (
                              <>
                                <p>✅ Query executed successfully</p>
                                <p>📊 Data retrieved and processed</p>
                                {stepResult?.data && Array.isArray(stepResult.data) && (
                                  <p>📋 Retrieved {stepResult.data.length} rows</p>
                                )}
                              </>
                            );
                          } else if (selectedStep?.name.includes('Visualization') || selectedStep?.name.includes('Chart')) {
                            return (
                              <>
                                <p>📈 Visualizations created successfully</p>
                                <p>🎨 Charts ready for display</p>
                              </>
                            );
                          } else {
                            return (
                              <>
                                <p>✅ {stepType} completed successfully</p>
                                <p>📊 Step processed and results generated</p>
                              </>
                            );
                          }
                        })()}
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
      {false && enhancedSteps.length > 0 && (
        <div className="plan-steps">
          <div className="steps-header" onClick={() => toggleStepExpansion(plan.plan_id)}>
            <p className="steps-title">
              <strong>📋 Processing Details ({enhancedSteps.length} steps)</strong>
              {!expandedSteps[plan.plan_id] && (
                <span className="steps-summary">
                  {' '}- {plan.status === 'completed' ? '✅ All steps completed' : 
                        plan.status === 'failed' ? '❌ Execution failed' :
                        `⚙️ ${progressPercentage}% complete`}
                </span>
              )}
            </p>
            <span className={`steps-toggle ${expandedSteps[plan.plan_id] ? 'expanded' : 'collapsed'}`}>
              {expandedSteps[plan.plan_id] ? '▼' : '▶'}
            </span>
          </div>
          
          {expandedSteps[plan.plan_id] && (
            <div className="steps-content">
              {enhancedSteps.map((step: PlanStep, index: number) => {
          // Use the step's output_data directly from enhanced steps
          const stepResult = step.output_data;
          
          console.log(`Step ${index + 1}: ${step.task_type} -> ${step.step_id}`, stepResult);
          
          return (
            <div key={`${step.task_type}_${index}`} className="plan-step">
              <div className={`step-status-icon ${step.status === 'error' ? 'failed' : 'completed'}`}>
                {step.status === 'error' ? '!' : '✓'}
              </div>
              <div className="step-details">
                <div className="step-name">
                  {index + 1}. {step.name}
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
                        
                        {/* Show data summary only (not full table) - full results are shown in dedicated section below */}
                        {(stepResult.data || stepResult.results) && Array.isArray(stepResult.data || stepResult.results) && (stepResult.data || stepResult.results).length > 0 && (
                          <div>
                            <p><strong>Query Results:</strong> {(stepResult.data || stepResult.results).length} rows retrieved successfully</p>
                            <p style={{color: '#6b7280', fontSize: '13px', fontStyle: 'italic'}}>
                              📊 Full results and visualizations are displayed in the dedicated section below
                            </p>
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
    </div>
    );
  };

  // Chart customization handlers
  const handleChartTypeChange = (chartIndex: number, stepId: string, newType: string) => {
    const chartKey = `${stepId}_${chartIndex}`;
    setCustomChartTypes(prev => ({
      ...prev,
      [chartKey]: newType
    }));
  };

  const handleColorSchemeChange = (chartIndex: number, stepId: string, colors: string[]) => {
    const chartKey = `${stepId}_${chartIndex}`;
    setCustomColors(prev => ({
      ...prev,
      [chartKey]: colors
    }));
  };

  const handleDownloadChart = (chartIndex: number, stepId: string, format: 'png' | 'pdf' | 'svg') => {
    const chartKey = `${stepId}_${chartIndex}`;
    const chartElement = document.querySelector(`[data-chart-id="${chartKey}"]`) as HTMLElement;
    
    if (chartElement) {
      const filename = `pharma_chart_${stepId}_${chartIndex}`;
      
      switch (format) {
        case 'png':
          downloadChartAsPNG(chartElement, filename);
          break;
        case 'pdf':
          downloadChartAsPDF(chartElement, filename);
          break;
        case 'svg':
          downloadChartAsSVG(chartElement, filename);
          break;
      }
    } else {
      console.error('Chart element not found for download');
    }
  };

  const getCustomizedChartData = (chart: any, chartIndex: number, stepId: string) => {
    const chartKey = `${stepId}_${chartIndex}`;
    const customType = customChartTypes[chartKey];
    const customColorScheme = customColors[chartKey];
    
    let customizedChart = chart;
    
    // Apply custom colors if available
    if (customColorScheme && customColorScheme.length > 0) {
      customizedChart = applyChartColors(customizedChart, customColorScheme);
    }
    
    // Apply custom chart type if available
    if (customType && customType !== chart.type) {
      customizedChart = convertChartType(customizedChart, customType);
    }
    
    return customizedChart;
  };

  // Render message
  const renderMessage = (message: ChatMessage) => {
    // Check if this assistant message has stored results
    const messageResult = messageResults[message.message_id];
    
    // For the latest message, also check current activePlan as fallback
    const isLatestMessage = message.message_id === messages[messages.length - 1]?.message_id;
    const fallbackPlan = isLatestMessage && activePlan && activePlan.status === 'completed' ? activePlan : null;
    
    const shouldShowResults = message.message_type === 'system_response' && (
      messageResult || 
      (isLatestMessage && fallbackPlan)
    );
    // Note: Removed the global activePlan check to prevent showing results on all messages

    // Debug logging
    if (message.message_type === 'system_response') {
      console.log('Rendering assistant message:', message.message_id, {
        messageResult: messageResult,
        fallbackPlan: fallbackPlan,
        shouldShowResults: shouldShowResults,
        isLatestMessage: isLatestMessage,
        activePlan: activePlan,
        allMessageResults: messageResults
      });
    }

    return (
      <div key={message.message_id} className={`message ${message.message_type === 'user_query' ? 'user' : 'assistant'}`}>
        <div className="message-avatar">
          {message.message_type === 'user_query' ? 
            (userProfile?.full_name.charAt(0) || 'U') : 'AI'
          }
        </div>
        <div className="message-content">
          <div className="message-bubble">
            {message.message_type === 'system_response' ? (
              <ReactMarkdown
                components={{
                  // Custom styling for markdown elements
                  p: ({children}) => <p style={{margin: '0 0 8px 0'}}>{children}</p>,
                  ul: ({children}) => <ul style={{margin: '8px 0', paddingLeft: '20px'}}>{children}</ul>,
                  li: ({children}) => <li style={{margin: '4px 0'}}>{children}</li>,
                  strong: ({children}) => <strong style={{color: '#1e40af', fontWeight: '600'}}>{children}</strong>,
                  h1: ({children}) => <h1 style={{fontSize: '1.25rem', fontWeight: '600', margin: '16px 0 8px 0', color: '#1f2937'}}>{children}</h1>,
                  h2: ({children}) => <h2 style={{fontSize: '1.1rem', fontWeight: '600', margin: '12px 0 6px 0', color: '#374151'}}>{children}</h2>,
                  h3: ({children}) => <h3 style={{fontSize: '1rem', fontWeight: '600', margin: '8px 0 4px 0', color: '#4b5563'}}>{children}</h3>
                }}
              >
                {message.content}
              </ReactMarkdown>
            ) : (
              message.content
            )}
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
          
          {/* Attach results directly to assistant messages */}
          {shouldShowResults && (
            <div className="message-results" style={{
              marginTop: '16px',
              background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(16, 185, 129, 0.05) 100%)',
              border: '1px solid rgba(59, 130, 246, 0.1)',
              borderRadius: '12px',
              padding: '20px',
              boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)'
            }}>
              {(() => {
                // Use stored results or fallback to current activePlan
                const planData = messageResult || fallbackPlan || (activePlan && activePlan.status === 'completed' ? activePlan : null);
                const resultsData = messageResult ? messageResult.results : 
                                  fallbackPlan ? fallbackPlan.results : 
                                  (activePlan && activePlan.status === 'completed' ? activePlan.results : {});
                
                // Get both visualization and execution results
                // The results are stored with keys like "2_execution", "3_visualization" etc.
                console.log('Available resultsData keys:', Object.keys(resultsData || {}));
                console.log('Full resultsData:', resultsData);
                
                const executionStep = Object.entries(resultsData || {}).find(([key, result]: [string, any]) => {
                  console.log(`Checking key: ${key}, includes execution: ${key.includes('execution')}, has results property: ${!!(result?.results)}, results length: ${result?.results?.length || 0}, status: ${result?.status}`);
                  return key.includes('execution') && result?.results !== undefined; // Just check if results property exists, not if it has data
                });
                
                const visualizationStep = Object.entries(resultsData || {}).find(([key, result]: [string, any]) => 
                  key.includes('visualization') && result?.charts !== undefined
                );

                const insightsStep = Object.entries(resultsData || {}).find(([key, result]: [string, any]) => 
                  key.includes('insights') && result?.insights !== undefined
                );

                // Debug visualization data
                console.log('🎨 Visualization debug:', {
                  visualizationStepFound: !!visualizationStep,
                  allSteps: Object.keys(resultsData || {}),
                  visualizationData: visualizationStep ? visualizationStep[1] : null,
                  rawVisualizationStep: resultsData?.['8_visualization_builder'],
                  chartsFound: visualizationStep ? (visualizationStep[1] as any)?.charts?.length : 0,
                  chartTypes: visualizationStep ? (visualizationStep[1] as any)?.charts?.map((c: any) => c.type) : []
                });

                const executionResult = executionStep ? executionStep[1] as any : null;
                const visualizationResult = visualizationStep ? visualizationStep[1] as any : null;
                const insightsResult = insightsStep ? insightsStep[1] as any : null;

                // Also check if any step has results, not just execution
                let allResults: any[] = [];
                let allCharts: any[] = [];
                let sqlQuery = '';
                
                Object.entries(resultsData || {}).forEach(([key, result]: [string, any]) => {
                  if (result?.results && Array.isArray(result.results)) {
                    allResults = allResults.concat(result.results);
                  }
                  if (result?.charts && Array.isArray(result.charts)) {
                    allCharts = allCharts.concat(result.charts);
                  }
                  if (result?.sql_query) {
                    sqlQuery = result.sql_query;
                  }
                });

                // Combine both results - try multiple strategies
                const stepResult: any = {
                  results: executionResult?.results || allResults || [],
                  sql_query: executionResult?.sql_query || sqlQuery || '',
                  charts: visualizationResult?.charts || allCharts || []
                };

                console.log('🎨 Chart extraction debug:', {
                  visualizationResult: visualizationResult,
                  hasVisualizationCharts: !!(visualizationResult?.charts),
                  visualizationChartsLength: visualizationResult?.charts?.length || 0,
                  allCharts: allCharts,
                  finalChartsLength: stepResult.charts.length,
                  finalCharts: stepResult.charts,
                  detailedChartStructure: stepResult.charts.map((chart: any, idx: number) => ({
                    index: idx,
                    type: chart.type,
                    hasData: !!chart.data,
                    dataKeys: chart.data ? Object.keys(chart.data) : null,
                    x_column: chart.x_column,
                    y_column: chart.y_column,
                    title: chart.title,
                    fullChart: chart
                  }))
                });

                // Create fallback chart ONLY if no visualization pipeline worked
                if (stepResult.results.length > 0 && stepResult.charts.length === 0) {
                  console.log('🎨 No charts found, creating fallback chart from execution data');
                  // Try to create a simple chart from the results
                  const results = stepResult.results;
                  if (results.length > 0) {
                    // Look for frequency/count-like data to chart
                    const firstRow = results[0];
                    const columns = Object.keys(firstRow);
                    
                    // Check if this looks like chart-able data
                    const hasFrequencyData = columns.some(col => 
                      col.toLowerCase().includes('frequency') || 
                      col.toLowerCase().includes('count') ||
                      col.toLowerCase().includes('total')
                    );
                    
                    if (hasFrequencyData) {
                      const fallbackChart = {
                        type: 'bar',
                        title: 'Data Visualization',
                        data: results,
                        x_column: columns.find(col => !col.toLowerCase().includes('frequency') && !col.toLowerCase().includes('count')) || columns[0],
                        y_column: columns.find(col => col.toLowerCase().includes('frequency') || col.toLowerCase().includes('count')) || columns[1]
                      };
                      stepResult.charts = [fallbackChart];
                      console.log('🎨 Created fallback chart:', fallbackChart);
                    }
                  }
                } else if (stepResult.charts.length > 0) {
                  console.log('🎨 Using provided charts from visualization pipeline:', stepResult.charts.length);
                }

                // Debug: Log the SQL queries to identify the issue
                console.log('🔍 SQL Query Debug:', {
                  generatedQuery: resultsData?.['2_query_generation']?.sql_query,
                  executionQuery: executionResult?.sql_query,
                  finalQuery: stepResult.sql_query,
                  executionResults: executionResult?.results?.length,
                  executionStatus: resultsData?.['3_execution']?.status
                });

                console.log('Rendering results for message:', message.message_id, {
                  hasMessageResult: !!messageResult,
                  hasFallbackPlan: !!fallbackPlan,
                  hasActivePlan: !!activePlan,
                  resultsData: resultsData,
                  executionStep: executionStep,
                  visualizationStep: visualizationStep,
                  executionResult: executionResult,
                  visualizationResult: visualizationResult,
                  stepResult: stepResult,
                  planData: planData
                });

                return (
                  <div>
                    {/* Debug info - minimal version */}
                    {planData && process.env.NODE_ENV === 'development' && (
                      <div style={{ 
                        background: '#f3f4f6', 
                        padding: '8px', 
                        fontSize: '11px', 
                        fontFamily: 'monospace',
                        marginBottom: '12px',
                        borderRadius: '4px'
                      }}>
                        Debug: Charts({stepResult.charts?.length || 0}) | Data({stepResult.results?.length || 0}) | SQL({stepResult.sql_query ? 'Yes' : 'No'}) | PlanData({!!planData ? 'Yes' : 'No'})
                      </div>
                    )}
                    
                    {/* Force debugging - can be removed in production */}
                    {false && planData && (
                      <div style={{ 
                        background: '#fff3cd', 
                        padding: '8px', 
                        fontSize: '11px', 
                        fontFamily: 'monospace',
                        marginBottom: '12px',
                        borderRadius: '4px'
                      }}>
                        FORCE DEBUG: resultsData keys: {Object.keys(resultsData || {}).join(', ')}
                        <br/>
                        FORCE DEBUG: stepResult.charts: {JSON.stringify(stepResult.charts, null, 2)}
                        <br/>
                        FORCE DEBUG: visualizationStep: {visualizationStep?.[0] ? `Found: ${visualizationStep?.[0]}` : 'Not found'}
                        <br/>
                        FORCE DEBUG: visualizationResult: {JSON.stringify(visualizationResult, null, 2)}
                        <br/>
                        FORCE DEBUG: allCharts length: {allCharts.length}
                        <br/>
                        FORCE DEBUG: stepResult.charts length: {stepResult.charts?.length || 0}
                      </div>
                    )}
                    
                    {/* Insights Display - Show textual insights when available */}
                    {insightsResult && insightsResult.insights && (
                      <div style={{
                        background: 'rgba(139, 69, 19, 0.05)',
                        border: '1px solid rgba(139, 69, 19, 0.2)',
                        borderRadius: '8px',
                        padding: '20px',
                        marginBottom: '16px'
                      }}>
                        <h5 style={{
                          margin: '0 0 16px 0',
                          fontSize: '16px',
                          fontWeight: '600',
                          color: '#8B4513',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '8px'
                        }}>
                          🧠 Generated Insights
                        </h5>
                        <div style={{
                          lineHeight: '1.6',
                          color: '#374151'
                        }}>
                          <ReactMarkdown
                            components={{
                              p: ({children}) => <p style={{margin: '0 0 12px 0'}}>{children}</p>,
                              ul: ({children}) => <ul style={{margin: '12px 0', paddingLeft: '20px'}}>{children}</ul>,
                              li: ({children}) => <li style={{margin: '6px 0'}}>{children}</li>,
                              strong: ({children}) => <strong style={{color: '#8B4513', fontWeight: '600'}}>{children}</strong>,
                              h1: ({children}) => <h1 style={{fontSize: '1.2rem', fontWeight: '600', margin: '16px 0 8px 0', color: '#8B4513'}}>{children}</h1>,
                              h2: ({children}) => <h2 style={{fontSize: '1.1rem', fontWeight: '600', margin: '14px 0 6px 0', color: '#A0522D'}}>{children}</h2>,
                              h3: ({children}) => <h3 style={{fontSize: '1rem', fontWeight: '600', margin: '12px 0 4px 0', color: '#CD853F'}}>{children}</h3>
                            }}
                          >
                            {insightsResult.insights}
                          </ReactMarkdown>
                        </div>
                        {insightsResult.row_count && (
                          <div style={{
                            marginTop: '12px',
                            padding: '8px 12px',
                            background: 'rgba(139, 69, 19, 0.1)',
                            borderRadius: '4px',
                            fontSize: '12px',
                            color: '#8B4513'
                          }}>
                            📊 Analysis based on {insightsResult.row_count} data rows
                          </div>
                        )}
                      </div>
                    )}
                    
                    {/* Emergency data display - show any data found - DISABLED for cleaner UI */}
                    {false && Object.entries(resultsData || {}).map(([key, result]: [string, any]) => {
                      if (key.includes('execution') && result?.results !== undefined) {
                        return (
                          <div key={key} style={{
                            background: 'rgba(16, 185, 129, 0.05)',
                            border: '1px solid rgba(16, 185, 129, 0.2)',
                            borderRadius: '8px',
                            padding: '16px',
                            marginBottom: '12px'
                          }}>
                            <h5 style={{
                              margin: '0 0 12px 0',
                              fontSize: '14px',
                              fontWeight: '600',
                              color: '#1F2937'
                            }}>📋 Query Execution Results from {key}</h5>
                            
                            {/* Show metadata */}
                            {result.metadata && (
                              <div style={{ marginBottom: '12px', fontSize: '12px' }}>
                                <strong>Columns found:</strong> {result.metadata.columns?.join(', ') || 'None'}
                                <br />
                                <strong>Row count:</strong> {result.row_count || 0}
                                <br />
                                <strong>Execution time:</strong> {result.execution_time?.toFixed(2) || 0}ms
                                <br />
                                <strong>Status:</strong> {result.status || 'Unknown'}
                              </div>
                            )}
                            
                            {result.results && result.results.length > 0 ? (
                              <div style={{
                                borderRadius: '4px',
                                overflow: 'hidden',
                                border: '1px solid rgba(229, 231, 235, 0.8)',
                                maxHeight: '300px'
                              }}>
                                <EnhancedTable 
                                  data={result.results}
                                  title=""
                                  description=""
                                  maxHeight="300px"
                                />
                              </div>
                            ) : (
                              <div style={{
                                textAlign: 'center',
                                padding: '20px',
                                background: 'rgba(249, 250, 251, 0.5)',
                                borderRadius: '4px',
                                color: '#6B7280'
                              }}>
                                <div style={{ fontSize: '24px', marginBottom: '8px' }}>📄</div>
                                <p style={{ margin: 0, fontSize: '14px' }}>Query executed successfully but returned no data</p>
                                <p style={{ margin: '4px 0 0 0', fontSize: '12px' }}>
                                  The query conditions may be too restrictive or the table may not contain matching records.
                                </p>
                              </div>
                            )}
                          </div>
                        );
                      }
                      return null;
                    })}
                    
                    {/* Side-by-side layout for charts and tables - Only show when there's actual data */}
                    {(stepResult.charts?.length > 0 || stepResult.results?.length > 0) && (
                      <div className="results-grid-container" style={{ 
                        display: stepResult.charts?.length > 0 && stepResult.results?.length > 0 ? 'grid' : 'block',
                        gridTemplateColumns: stepResult.charts?.length > 0 && stepResult.results?.length > 0 ? '1fr 1fr' : '1fr',
                      }}>
                        
                        {/* Left side: Charts - Only show if there are actual charts */}
                        {stepResult.charts && stepResult.charts.length > 0 && (
                          <div className="chart-container">
                            <h5 className="section-header">
                              📊 Visualization ({stepResult.charts.length})
                            </h5>
                            <div className="chart-wrapper">
                              {stepResult.charts.map((chart: any, idx: number) => {
                                const stepId = messageResult?.plan?.plan_id || message.message_id;
                                const chartKey = `${stepId}_${idx}`;
                                
                                // Add detailed logging for each chart
                                console.log(`🎨 Rendering chart ${idx}:`, {
                                  chart: chart,
                                  type: chart.type,
                                  hasData: !!chart.data,
                                  dataStructure: chart.data ? Object.keys(chart.data) : 'No data',
                                  x_column: chart.x_column,
                                  y_column: chart.y_column,
                                  willRenderMatplotlib: chart.type === 'matplotlib' && chart.data,
                                  willRenderBar: chart.type === 'bar' && chart.data && chart.x_column && chart.y_column,
                                  willRenderPlotly: chart.type === 'plotly' && chart.data,
                                  chartDataSample: chart.data ? (Array.isArray(chart.data) ? chart.data.slice(0, 2) : chart.data) : null
                                });
                                
                                return (
                                  <div key={idx} style={{ 
                                    width: '100%', 
                                    height: '100%', 
                                    minHeight: '350px',
                                    display: 'flex',
                                    flexDirection: 'column',
                                    border: '2px dashed #e5e7eb', // Debug border to see if container is rendered
                                    position: 'relative'
                                  }}>
                                    {/* Debug indicator for chart processing */}
                                    <div style={{
                                      position: 'absolute',
                                      top: '8px',
                                      right: '8px',
                                      background: '#3b82f6',
                                      color: 'white',
                                      padding: '4px 8px',
                                      borderRadius: '4px',
                                      fontSize: '10px',
                                      zIndex: 1000
                                    }}>
                                      Chart #{idx} - Type: {chart.type}
                                    </div>
                                    {chart.type === 'matplotlib' && chart.data && (
                                      <div style={{ 
                                        width: '100%', 
                                        height: '100%',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center'
                                      }}>
                                        <img 
                                          src={chart.data} 
                                          alt={chart.title || 'Visualization'}
                                          style={{ 
                                            maxWidth: '100%', 
                                            maxHeight: '100%',
                                            width: 'auto',
                                            height: 'auto',
                                            objectFit: 'contain',
                                            borderRadius: '4px' 
                                          }}
                                        />
                                      </div>
                                    )}
                                    
                                    {chart.type === 'bar' && chart.data && chart.x_column && chart.y_column && (
                                      <div style={{ 
                                        width: '100%', 
                                        height: '350px',
                                        minWidth: '300px'
                                      }}>
                                        <Plot
                                          data={[{
                                            x: chart.data.map((row: any) => row[chart.x_column]),
                                            y: chart.data.map((row: any) => row[chart.y_column]),
                                            type: 'bar',
                                            marker: {
                                              color: 'rgba(55, 128, 191, 0.7)',
                                              line: {
                                                color: 'rgba(55, 128, 191, 1.0)',
                                                width: 1
                                              }
                                            }
                                          }]}
                                          layout={{
                                            title: chart.title || 'Data Visualization',
                                            autosize: true,
                                            width: undefined,
                                            height: undefined,
                                            margin: { t: 50, r: 30, b: 50, l: 60 },
                                            paper_bgcolor: 'rgba(255,255,255,0.9)',
                                            plot_bgcolor: 'rgba(255,255,255,0.9)',
                                            font: {
                                              size: 12,
                                              family: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif'
                                            },
                                            xaxis: {
                                              title: chart.x_column
                                            },
                                            yaxis: {
                                              title: chart.y_column
                                            }
                                          }}
                                          config={{
                                            displayModeBar: true,
                                            displaylogo: false,
                                            responsive: true,
                                            toImageButtonOptions: {
                                              format: 'png',
                                              filename: 'chart',
                                              height: 500,
                                              width: 700,
                                              scale: 1
                                            },
                                            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
                                          }}
                                          style={{
                                            width: '100%', 
                                            height: '100%'
                                          }}
                                          useResizeHandler={true}
                                        />
                                      </div>
                                    )}
                                    
                                    {chart.type === 'plotly' && chart.data && (
                                      <div style={{ 
                                        width: '100%', 
                                        height: '350px',
                                        minWidth: '300px'
                                      }}>
                                        {(() => {
                                          // Check if plotly data has actual data points
                                          const plotlyData = chart.data.data || [];
                                          const hasActualData = plotlyData.some((trace: any) => 
                                            trace.x && Array.isArray(trace.x) && trace.x.length > 0 ||
                                            trace.y && Array.isArray(trace.y) && trace.y.length > 0
                                          );
                                          
                                          console.log('🎨 Plotly chart data check:', {
                                            plotlyData: plotlyData,
                                            hasActualData: hasActualData,
                                            firstTrace: plotlyData[0]
                                          });
                                          
                                          if (!hasActualData) {
                                            return (
                                              <div style={{
                                                width: '100%',
                                                height: '100%',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                background: '#fef3c7',
                                                border: '2px dashed #f59e0b',
                                                borderRadius: '8px',
                                                flexDirection: 'column',
                                                gap: '12px'
                                              }}>
                                                <div style={{ fontSize: '24px' }}>📊</div>
                                                <div style={{ textAlign: 'center', color: '#92400e' }}>
                                                  <div style={{ fontWeight: '600', marginBottom: '4px' }}>
                                                    Chart Data Processing Error
                                                  </div>
                                                  <div style={{ fontSize: '12px' }}>
                                                    The backend generated a chart but the data extraction failed.
                                                    <br/>
                                                    This is likely due to unsafe eval() usage in the Python code.
                                                    <br/>
                                                    The backend will be fixed to use safe JSON parsing instead.
                                                  </div>
                                                </div>
                                              </div>
                                            );
                                          }
                                          
                                          return (
                                            <Plot
                                              data={chart.data.data || []}
                                              layout={{
                                                ...(chart.data.layout || {}),
                                                autosize: true,
                                                width: undefined,
                                                height: undefined,
                                                margin: { t: 50, r: 30, b: 50, l: 60 },
                                                paper_bgcolor: 'rgba(255,255,255,0.9)',
                                                plot_bgcolor: 'rgba(255,255,255,0.9)',
                                                font: {
                                                  size: 12,
                                                  family: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif'
                                                },
                                                showlegend: true,
                                                legend: {
                                                  orientation: 'v',
                                                  x: 1.02,
                                                  y: 1
                                                }
                                              }}
                                              config={{
                                                displayModeBar: true,
                                                displaylogo: false,
                                                responsive: true,
                                                toImageButtonOptions: {
                                                  format: 'png',
                                                  filename: 'chart',
                                                  height: 500,
                                                  width: 700,
                                                  scale: 1
                                                },
                                                modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
                                              }}
                                              style={{
                                                width: '100%', 
                                                height: '100%'
                                              }}
                                              useResizeHandler={true}
                                              onInitialized={(figure: any, graphDiv: any) => {
                                                // Force resize after initialization
                                                setTimeout(() => {
                                                  if (window.Plotly && graphDiv) {
                                                    window.Plotly.Plots.resize(graphDiv);
                                                  }
                                                }, 100);
                                              }}
                                            />
                                          );
                                        })()}
                                      </div>
                                    )}
                                    
                                    {/* Catch-all debug section for unhandled chart types */}
                                    {chart.type !== 'matplotlib' && chart.type !== 'bar' && chart.type !== 'plotly' && (
                                      <div style={{
                                        background: '#fef3c7',
                                        border: '1px solid #f59e0b',
                                        borderRadius: '8px',
                                        padding: '16px',
                                        margin: '8px 0'
                                      }}>
                                        <h6 style={{ margin: '0 0 8px 0', color: '#92400e' }}>
                                          🐛 DEBUG: Unhandled Chart Type
                                        </h6>
                                        <pre style={{ 
                                          fontSize: '10px',
                                          margin: 0,
                                          whiteSpace: 'pre-wrap',
                                          maxHeight: '200px',
                                          overflow: 'auto'
                                        }}>
                                          {JSON.stringify(chart, null, 2)}
                                        </pre>
                                      </div>
                                    )}
                                    
                                    {/* Debug section for charts with missing data */}
                                    {(chart.type === 'bar' && (!chart.data || !chart.x_column || !chart.y_column)) && (
                                      <div style={{
                                        background: '#fecaca',
                                        border: '1px solid #ef4444',
                                        borderRadius: '8px',
                                        padding: '16px',
                                        margin: '8px 0'
                                      }}>
                                        <h6 style={{ margin: '0 0 8px 0', color: '#991b1b' }}>
                                          ❌ DEBUG: Bar Chart Missing Required Data
                                        </h6>
                                        <p style={{ fontSize: '11px', margin: '4px 0' }}>
                                          Has data: {!!chart.data ? 'Yes' : 'No'}<br/>
                                          Has x_column: {!!chart.x_column ? `Yes (${chart.x_column})` : 'No'}<br/>
                                          Has y_column: {!!chart.y_column ? `Yes (${chart.y_column})` : 'No'}
                                        </p>
                                      </div>
                                    )}
                                    
                                    {/* Debug section for plotly charts with missing data */}
                                    {(chart.type === 'plotly' && !chart.data) && (
                                      <div style={{
                                        background: '#fecaca',
                                        border: '1px solid #ef4444',
                                        borderRadius: '8px',
                                        padding: '16px',
                                        margin: '8px 0'
                                      }}>
                                        <h6 style={{ margin: '0 0 8px 0', color: '#991b1b' }}>
                                          ❌ DEBUG: Plotly Chart Missing Data
                                        </h6>
                                        <pre style={{ 
                                          fontSize: '10px',
                                          margin: 0,
                                          whiteSpace: 'pre-wrap',
                                          maxHeight: '200px',
                                          overflow: 'auto'
                                        }}>
                                          {JSON.stringify(chart, null, 2)}
                                        </pre>
                                      </div>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}

                        {/* Right side: Data Table - Only show if there are actual results */}
                        {stepResult.results && stepResult.results.length > 0 && (
                          <div className="data-container">
                            <h5 className="section-header">
                              📋 Data ({stepResult.results.length} rows)
                            </h5>
                            <div className="data-table-wrapper">
                              <EnhancedTable 
                                data={stepResult.results}
                                title=""
                                description=""
                                maxHeight="320px"
                              />
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* AI Agent Execution Plan - Enhanced Card Format */}
                    {planData && (() => {
                      const actualPlan = planData.plan || planData;
                      const hasDbOperations = actualPlan.tasks && actualPlan.tasks.some((task: any) => 
                        task.task_type && ['schema_discovery', 'query_generation', 'execution', 'sql_generation', 'python_generation', 'visualization_builder'].includes(task.task_type)
                      );
                      const hasExecutableSteps = actualPlan.reasoning_steps && actualPlan.reasoning_steps.some((step: string) => 
                        step.toLowerCase().includes('sql') || 
                        step.toLowerCase().includes('query') || 
                        step.toLowerCase().includes('database') ||
                        step.toLowerCase().includes('table')
                      );
                      
                      return (hasDbOperations || hasExecutableSteps);
                    })() && (
                      <div style={{
                        background: 'rgba(99, 102, 241, 0.05)',
                        border: '1px solid rgba(99, 102, 241, 0.2)',
                        borderRadius: '12px',
                        padding: '20px',
                        marginTop: '16px'
                      }}>
                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          marginBottom: '16px'
                        }}>
                          <h5 style={{
                            margin: 0,
                            fontSize: '16px',
                            fontWeight: '600',
                            color: '#1F2937',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px'
                          }}>🤖 AI Agent Execution Plan</h5>
                        </div>
                        
                        {/* Get the actual plan object */}
                        {(() => {
                          const actualPlan = planData.plan || planData;
                          return (
                            <>
                              {/* Reasoning Process */}
                              {actualPlan.reasoning_steps && (
                                <div style={{ marginBottom: '16px' }}>
                                  <div style={{
                                    fontSize: '12px',
                                    fontWeight: '500',
                                    color: '#6366F1',
                                    marginBottom: '8px'
                                  }}>Reasoning Process:</div>
                                  <ul style={{
                                    margin: 0,
                                    paddingLeft: '16px',
                                    fontSize: '12px',
                                    lineHeight: '1.5',
                                    color: '#4B5563'
                                  }}>
                                    {actualPlan.reasoning_steps.map((step: string, idx: number) => (
                                      <li key={idx} style={{ marginBottom: '4px' }}>{step}</li>
                                    ))}
                                  </ul>
                                </div>
                              )}

                              {/* Execution Steps - Enhanced Interactive Card Layout */}
                              {actualPlan.tasks && actualPlan.tasks.length > 0 && (
                                <div>
                                  <div style={{
                                    fontSize: '12px',
                                    fontWeight: '500',
                                    color: '#6366F1',
                                    marginBottom: '12px'
                                  }}>Execution Steps ({actualPlan.tasks.length}):</div>
                                  
                                  {/* Visual Step Cards */}
                                  <div style={{
                                    display: 'flex',
                                    gap: '12px',
                                    flexWrap: 'wrap',
                                    marginBottom: '16px'
                                  }}>
                                    {actualPlan.tasks.map((task: any, idx: number) => {
                                      const taskName = task.task_type?.replace(/_/g, ' ') || task.agent;
                                      const displayName = taskName.replace(/\b\w/g, (l: string) => l.toUpperCase());
                                      const stepKey = `step_${idx}`;
                                      const isSelected = selectedStepKey === stepKey;
                                      
                                      return (
                                        <div 
                                          key={idx} 
                                          style={{
                                            display: 'flex',
                                            flexDirection: 'column',
                                            alignItems: 'center',
                                            minWidth: '120px',
                                            cursor: 'pointer',
                                            transform: isSelected ? 'scale(1.05)' : 'scale(1)',
                                            transition: 'all 0.2s ease'
                                          }}
                                          onClick={() => setSelectedStepKey(isSelected ? null : stepKey)}
                                        >
                                          {/* Step Circle */}
                                          <div style={{
                                            width: '40px',
                                            height: '40px',
                                            borderRadius: '50%',
                                            background: isSelected ? '#6366F1' : '#10B981',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            marginBottom: '8px',
                                            transition: 'all 0.3s ease',
                                            border: isSelected ? '2px solid #4F46E5' : '2px solid transparent',
                                            boxShadow: isSelected ? '0 4px 12px rgba(99, 102, 241, 0.3)' : 'none'
                                          }}>
                                            <FiCheck style={{ color: 'white', fontSize: '18px' }} />
                                          </div>
                                          
                                          {/* Step Label */}
                                          <div style={{
                                            fontSize: '12px',
                                            fontWeight: '500',
                                            color: isSelected ? '#6366F1' : '#10B981',
                                            textAlign: 'center',
                                            lineHeight: '1.3'
                                          }}>
                                            {displayName}
                                          </div>
                                        </div>
                                      );
                                    })}
                                  </div>
                                  
                                  {/* Step Details Panel */}
                                  {selectedStepKey && (() => {
                                    const stepIndex = parseInt(selectedStepKey.replace('step_', ''));
                                    const selectedTask = actualPlan.tasks[stepIndex];
                                    const taskType = selectedTask?.task_type;
                                    
                                    // Find the corresponding result data
                                    const stepResults = actualPlan.results || messageResult?.results || {};
                                    let stepData = null;
                                    
                                    // Try to find step data by task type or index
                                    if (stepResults[taskType]) {
                                      stepData = stepResults[taskType];
                                    } else if (stepResults[`${stepIndex + 1}_${taskType}`]) {
                                      stepData = stepResults[`${stepIndex + 1}_${taskType}`];
                                    } else {
                                      // Look for any matching results
                                      Object.keys(stepResults).forEach(key => {
                                        if (key.includes(taskType) || key.includes(String(stepIndex + 1))) {
                                          stepData = stepResults[key];
                                        }
                                      });
                                    }
                                    
                                    return (
                                      <div style={{
                                        background: 'rgba(99, 102, 241, 0.1)',
                                        border: '1px solid rgba(99, 102, 241, 0.2)',
                                        borderRadius: '8px',
                                        padding: '16px',
                                        marginTop: '8px'
                                      }}>
                                        <div style={{
                                          fontSize: '14px',
                                          fontWeight: '600',
                                          color: '#4F46E5',
                                          marginBottom: '12px',
                                          display: 'flex',
                                          alignItems: 'center',
                                          gap: '8px'
                                        }}>
                                          <FiInfo size={16} />
                                          {selectedTask?.task_type?.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())} Details
                                        </div>
                                        
                                        {/* Show step-specific content */}
                                        {(() => {
                                          const taskType = selectedTask?.task_type;
                                          
                                          // Query Generation Step - Show SQL
                                          if (taskType === 'query_generation' && stepData?.sql_query) {
                                            return (
                                              <div>
                                                <div style={{
                                                  fontSize: '12px',
                                                  fontWeight: '500',
                                                  color: '#6366F1',
                                                  marginBottom: '8px'
                                                }}>Generated SQL Query:</div>
                                                <pre style={{
                                                  background: 'rgba(15, 23, 42, 0.95)',
                                                  color: '#E2E8F0',
                                                  padding: '12px',
                                                  borderRadius: '6px',
                                                  fontSize: '11px',
                                                  lineHeight: '1.4',
                                                  overflow: 'auto',
                                                  fontFamily: 'Monaco, Consolas, monospace',
                                                  margin: 0
                                                }}>{stepData.sql_query}</pre>
                                              </div>
                                            );
                                          }
                                          
                                          // Python Generation Step - Show Python Code
                                          if (taskType === 'python_generation' && stepData?.python_code) {
                                            return (
                                              <div>
                                                <div style={{
                                                  fontSize: '12px',
                                                  fontWeight: '500',
                                                  color: '#6366F1',
                                                  marginBottom: '8px'
                                                }}>Generated Python Code:</div>
                                                <pre style={{
                                                  background: 'rgba(15, 23, 42, 0.95)',
                                                  color: '#E2E8F0',
                                                  padding: '12px',
                                                  borderRadius: '6px',
                                                  fontSize: '11px',
                                                  lineHeight: '1.4',
                                                  overflow: 'auto',
                                                  fontFamily: 'Monaco, Consolas, monospace',
                                                  margin: 0,
                                                  maxHeight: '300px'
                                                }}>{stepData.python_code}</pre>
                                              </div>
                                            );
                                          }
                                          
                                          // Schema Discovery - Show Tables
                                          if (taskType === 'schema_discovery' && stepData?.discovered_tables) {
                                            return (
                                              <div>
                                                <div style={{
                                                  fontSize: '12px',
                                                  fontWeight: '500',
                                                  color: '#6366F1',
                                                  marginBottom: '8px'
                                                }}>Discovered Tables ({stepData.discovered_tables.length}):</div>
                                                <div style={{
                                                  display: 'flex',
                                                  flexWrap: 'wrap',
                                                  gap: '6px'
                                                }}>
                                                  {stepData.discovered_tables.map((table: string, i: number) => (
                                                    <span key={i} style={{
                                                      background: 'rgba(59, 130, 246, 0.1)',
                                                      color: '#1E40AF',
                                                      padding: '4px 8px',
                                                      borderRadius: '4px',
                                                      fontSize: '11px'
                                                    }}>
                                                      {table}
                                                    </span>
                                                  ))}
                                                </div>
                                              </div>
                                            );
                                          }
                                          
                                          // Execution Step - Show Data Summary
                                          if (taskType === 'execution' && stepData?.results) {
                                            return (
                                              <div>
                                                <div style={{
                                                  fontSize: '12px',
                                                  fontWeight: '500',
                                                  color: '#6366F1',
                                                  marginBottom: '8px'
                                                }}>Execution Summary:</div>
                                                <div style={{
                                                  fontSize: '12px',
                                                  color: '#4B5563'
                                                }}>
                                                  <p style={{ margin: '4px 0' }}>✅ Query executed successfully</p>
                                                  <p style={{ margin: '4px 0' }}>📊 Retrieved {stepData.results.length} rows</p>
                                                  {stepData.results.length > 0 && (
                                                    <p style={{ margin: '4px 0' }}>🔍 Columns: {Object.keys(stepData.results[0]).join(', ')}</p>
                                                  )}
                                                </div>
                                              </div>
                                            );
                                          }
                                          
                                          // Visualization Builder Step - Show Charts Info
                                          if (taskType === 'visualization_builder' && stepData?.charts) {
                                            return (
                                              <div>
                                                <div style={{
                                                  fontSize: '12px',
                                                  fontWeight: '500',
                                                  color: '#6366F1',
                                                  marginBottom: '8px'
                                                }}>Visualization Summary:</div>
                                                <div style={{
                                                  fontSize: '12px',
                                                  color: '#4B5563'
                                                }}>
                                                  <p style={{ margin: '4px 0' }}>📊 Generated {stepData.charts.length} visualization(s)</p>
                                                  {stepData.charts.map((chart: any, i: number) => (
                                                    <p key={i} style={{ margin: '4px 0' }}>
                                                      • {chart.type || 'Chart'} - {chart.title || `Chart ${i + 1}`}
                                                    </p>
                                                  ))}
                                                </div>
                                              </div>
                                            );
                                          }
                                          
                                          // Default - Show Summary
                                          return (
                                            <div style={{
                                              fontSize: '12px',
                                              color: '#4B5563',
                                              fontStyle: 'italic'
                                            }}>
                                              {stepData ? 
                                                `Step completed successfully. Click other steps to see their details.` : 
                                                `Step completed. No detailed output available for this step type.`
                                              }
                                            </div>
                                          );
                                        })()}
                                      </div>
                                    );
                                  })()}
                                </div>
                              )}

                              {/* Plan Metadata */}
                              {(actualPlan.estimated_execution_time || actualPlan.plan_id) && (
                                <div style={{
                                  paddingTop: '12px',
                                  borderTop: '1px solid rgba(99, 102, 241, 0.1)',
                                  fontSize: '12px',
                                  color: '#6B7280',
                                  display: 'flex',
                                  justifyContent: 'space-between',
                                  alignItems: 'center'
                                }}>
                                  <div>
                                    <span>🔄 Analyzed database structure and content</span>
                                  </div>
                                  <div style={{ display: 'flex', gap: '16px' }}>
                                    {actualPlan.estimated_execution_time && (
                                      <span>⏱️ Est. Time: {actualPlan.estimated_execution_time}</span>
                                    )}
                                    {actualPlan.plan_id && (
                                      <span>🆔 Plan ID: {actualPlan.plan_id.substring(0, 8)}...</span>
                                    )}
                                  </div>
                                </div>
                              )}
                            </>
                          );
                        })()}
                      </div>
                    )}

                    {/* SQL Query - Hidden by default */}
                    {false && stepResult.sql_query && (
                      <div style={{
                        background: 'rgba(15, 23, 42, 0.95)',
                        borderRadius: '6px',
                        padding: '12px',
                        marginTop: '8px'
                      }}>
                        <div style={{
                          fontSize: '12px',
                          fontWeight: '500',
                          color: '#94A3B8',
                          marginBottom: '8px'
                        }}>Generated SQL Query:</div>
                        <pre style={{
                          margin: 0,
                          color: '#E2E8F0',
                          fontSize: '11px',
                          lineHeight: '1.4',
                          overflow: 'auto',
                          fontFamily: 'Monaco, Consolas, monospace'
                        }}>{stepResult.sql_query}</pre>
                      </div>
                    )}
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      </div>
    );
  };

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
            <div className="chat-history-buttons">
              <button 
                className={`clear-chat-btn ${!activeConversation ? 'disabled' : ''}`}
                onClick={handleClearChat}
                disabled={!activeConversation}
                title="Clear current chat"
              >
                <FiArchive size={12} /> Clear
              </button>
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
                    {(() => {
                      const activity = formatRecentActivity(conversation.last_activity_timestamp || Date.now() - 86400000);
                      return activity.text && (
                        <span 
                          className="activity-indicator" 
                          style={{ color: activity.color }}
                        >
                          <FiZap size={10} />
                        </span>
                      );
                    })()}
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
            <button className="action-btn" onClick={() => setShowQuickActions(!showQuickActions)}>
              <FiZap size={14} /> Quick Actions
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

        {/* Quick Actions Panel */}
        {showQuickActions && (
          <div className="quick-actions-panel">
            <div className="quick-actions-header">
              <h3>Quick Marketing Actions</h3>
              <button 
                className="close-btn" 
                onClick={() => setShowQuickActions(false)}
              >
                ×
              </button>
            </div>
            <div className="quick-actions-grid">
              {quickActions.map((action, index) => (
                <button
                  key={index}
                  className="quick-action-item"
                  onClick={() => {
                    setCurrentMessage(action);
                    setShowQuickActions(false);
                  }}
                >
                  <div className="action-icon">
                    <FiZap size={16} />
                  </div>
                  <div className="action-content">
                    <div className="action-title">Marketing Query {index + 1}</div>
                    <div className="action-description">{action}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Messages */}
        <div className="messages-container">
          {messages.map((message, index) => {
            // Check if this user message is being processed (no assistant response after it)
            const isUserMessageBeingProcessed = message.message_type === 'user_query' && 
                                              isLoading &&
                                              (index === messages.length - 1 || 
                                               (index < messages.length - 1 && messages[index + 1].message_type === 'user_query'));
            
            return (
              <div key={message.message_id}>
                {renderMessage(message)}
                
                {/* Show progress indicator right after user query being processed */}
                {isUserMessageBeingProcessed && progressSteps.length > 0 && (
                  <div 
                    key={`progress-${message.message_id}`}
                    style={{ 
                      margin: '16px 0', 
                      padding: '0 60px',
                      animation: 'slideIn 0.3s ease-out'
                    }}
                  >
                    {/* AI Agent Execution Plan Card - replacing ProgressIndicator */}
                    <div style={{
                      background: 'rgba(99, 102, 241, 0.05)',
                      border: '1px solid rgba(99, 102, 241, 0.2)',
                      borderRadius: '12px',
                      padding: '20px',
                      marginBottom: '16px',
                      animation: 'slideIn 0.3s ease-out'
                    }}>
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        marginBottom: '16px'
                      }}>
                        <h5 style={{
                          margin: 0,
                          fontSize: '16px',
                          fontWeight: '600',
                          color: '#1F2937',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '8px'
                        }}>🤖 AI Agent Execution Plan</h5>
                      </div>

                      {/* Progress Steps as Cards */}
                      <div style={{
                        display: 'flex',
                        gap: '12px',
                        flexWrap: 'wrap',
                        marginBottom: '16px'
                      }}>
                        {progressSteps.map((step, index) => {
                          const isCompleted = step.status === 'completed';
                          const isRunning = step.status === 'running';
                          const isPending = step.status === 'pending';
                          
                          return (
                            <div key={step.id} style={{
                              display: 'flex',
                              flexDirection: 'column',
                              alignItems: 'center',
                              minWidth: '120px'
                            }}>
                              {/* Step Circle */}
                              <div style={{
                                width: '40px',
                                height: '40px',
                                borderRadius: '50%',
                                background: isCompleted ? '#10B981' : isRunning ? '#F59E0B' : '#E5E7EB',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                marginBottom: '8px',
                                transition: 'all 0.3s ease'
                              }}>
                                {isCompleted ? (
                                  <FiCheck style={{ color: 'white', fontSize: '18px' }} />
                                ) : isRunning ? (
                                  <FiLoader style={{ color: 'white', fontSize: '16px', animation: 'spin 1s linear infinite' }} />
                                ) : (
                                  <span style={{ color: '#9CA3AF', fontSize: '14px', fontWeight: '600' }}>
                                    {index + 1}
                                  </span>
                                )}
                              </div>
                              
                              {/* Step Label */}
                              <div style={{
                                fontSize: '12px',
                                fontWeight: '500',
                                color: isCompleted ? '#10B981' : isRunning ? '#F59E0B' : '#6B7280',
                                textAlign: 'center',
                                lineHeight: '1.3'
                              }}>
                                {step.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      {/* Execution Summary */}
                      <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        paddingTop: '12px',
                        borderTop: '1px solid rgba(99, 102, 241, 0.1)',
                        fontSize: '12px',
                        color: '#6B7280'
                      }}>
                        <div>
                          <span>🔄 Planned {progressSteps.length} execution steps</span>
                        </div>
                        <div>
                          {estimatedTotalTime && (
                            <span>⏱️ Est. Time: {estimatedTotalTime}s</span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
          
          {/* Note: Execution plan is now shown inline with messages, not as a separate component */}
          
          {isLoading && (
            <div className="loading-indicator">
              <div className="loading-content">
                {/* Progress is now shown inline with messages above */}
                <div className="loading-state">
                  <div className="loading-content">
                    <div className="loading-spinner"></div>
                    <p>{activePlan ? 'Processing your query...' : 'Initializing analysis...'}</p>
                  </div>
                </div>
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
