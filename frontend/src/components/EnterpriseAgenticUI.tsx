import React, { useState, useEffect, useRef } from 'react';
import './EnterpriseAgenticUI.css';
import { FiSend, FiCopy, FiCheck, FiDatabase, FiSettings, FiBarChart2, FiTable, FiAlertTriangle, FiInfo, FiChevronDown, FiChevronUp, FiLoader } from 'react-icons/fi';
import { AdaptiveLayout, IntelligentVisualizationResult } from './visualizations';

// Type definitions
interface Message {
  id: string;
  type: 'user' | 'agent';
  text?: string;
  plan?: Plan;
}

interface Plan {
  plan_id: string;
  status: 'draft' | 'validated' | 'executing' | 'completed' | 'failed' | 'approved' | 'requires_approval';
  progress: number;
  reasoning_steps: string[];
  execution_steps: ExecutionStep[];
  estimated_cost?: number;
  actual_cost?: number;
  context: {
    generated_sql?: string;
    visualizations?: {
      charts?: Chart[];
    };
    data?: any[];
    query_results?: {
      data?: any[];
    };
    // NEW: Intelligent visualization planning result
    intelligent_visualization_planning?: IntelligentVisualizationResult;
  };
}

interface ExecutionStep {
  step_id: string;
  tool_type: string;
  status: string;
}

interface Chart {
  type: string;
  title: string;
  data: any;
}

// Mock API functions
const mockApi = {
  agentQuery: async (query: string, userId: string, sessionId: string) => {
    console.log("Sending query:", { query, userId, sessionId });
    // Simulate API call
    await new Promise(res => setTimeout(res, 1000));
    const planId = `plan_${Date.now()}`;
    return { plan_id: planId, status: 'draft' };
  },
  getPlanStatus: async (planId: string) => {
    console.log("Getting status for plan:", planId);
    // Simulate plan progress
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
  }
};


const EnterpriseAgenticUI = () => {
  const [query, setQuery] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [activePlan, setActivePlan] = useState<Plan | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (activePlan && activePlan.status !== 'completed' && activePlan.status !== 'failed') {
      interval = setInterval(async () => {
        const statusRes = await mockApi.getPlanStatus(activePlan.plan_id);
        
        setMessages(prev => prev.map(msg => 
          msg.id === activePlan.plan_id ? { ...msg, plan: statusRes as Plan } : msg
        ));

        if (statusRes.status === 'completed' || statusRes.status === 'failed') {
          setActivePlan(null);
          clearInterval(interval);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [activePlan]);

  const handleSendQuery = async () => {
    if (!query.trim()) return;

    const userMessage: Message = {
      id: `user_${Date.now()}`,
      type: 'user' as const,
      text: query,
    };
    setMessages(prev => [...prev, userMessage]);
    setQuery('');
    setIsLoading(true);

    try {
      const res = await mockApi.agentQuery(query, 'test_user', 'session_123');
      
      const agentMessage: Message = {
        id: res.plan_id,
        type: 'agent' as const,
        plan: {
          plan_id: res.plan_id,
          status: 'draft',
          progress: 0,
          reasoning_steps: [],
          execution_steps: [],
          context: {}
        }
      };
      
      setMessages(prev => [...prev, agentMessage]);
      setActivePlan({ 
        plan_id: res.plan_id, 
        status: 'draft' as const,
        progress: 0,
        reasoning_steps: [],
        execution_steps: [],
        context: {}
      });

    } catch (error) {
      const errorMessage: Message = {
        id: `error_${Date.now()}`,
        type: 'agent' as const,
        text: 'Failed to start agent query. Please try again.',
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="agentic-ui">
      <header className="agentic-header">
        <h1>Enterprise NL2Query Agent</h1>
        <div className="header-actions">
          <FiDatabase />
          <span>Connected to: Snowflake Prod</span>
          <button className="settings-btn"><FiSettings /> Settings</button>
        </div>
      </header>
      <main className="agentic-main">
        <div className="chat-container">
          <div className="message-list">
            {messages.map(msg => (
              <Message key={msg.id} message={msg} />
            ))}
            <div ref={messagesEndRef} />
          </div>
          <div className="chat-input-area">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !isLoading && handleSendQuery()}
              placeholder="Ask a question about your data..."
              disabled={isLoading}
            />
            <button onClick={handleSendQuery} disabled={isLoading}>
              {isLoading ? <FiLoader className="spin" /> : <FiSend />}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
};

const Message = ({ message }: { message: Message }) => {
  if (message.type === 'user') {
    return <div className="message user-message">{message.text}</div>;
  }
  if (message.type === 'agent' && message.text) {
    return <div className="message error-message"><FiAlertTriangle /> {message.text}</div>;
  }
  if (message.type === 'agent' && message.plan) {
    return <AgentMessage plan={message.plan} />;
  }
  return null;
};

const AgentMessage = ({ plan }: { plan: Plan }) => {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <div className="message agent-message">
      <div className="agent-message-content">
        <PlanStatusDisplay plan={plan} />
        {plan.status === 'completed' && <ResultsDisplay plan={plan} />}
        {plan.status === 'requires_approval' && <ApprovalPrompt planId={plan.plan_id} />}
        
        <div className="details-toggle" onClick={() => setShowDetails(!showDetails)}>
          {showDetails ? <FiChevronUp /> : <FiChevronDown />}
          <span>Execution Details</span>
        </div>

        {showDetails && <PlanDetails plan={plan} />}
      </div>
    </div>
  );
};

const PlanStatusDisplay = ({ plan }: { plan: Plan }) => {
  const getStatusIcon = () => {
    switch (plan.status) {
      case 'completed': return <FiCheck className="status-icon success" />;
      case 'failed': return <FiAlertTriangle className="status-icon error" />;
      case 'executing': return <FiLoader className="status-icon spin" />;
      default: return <FiInfo className="status-icon info" />;
    }
  };

  return (
    <div className="plan-status">
      <div className="status-header">
        {getStatusIcon()}
        <span>Status: {plan.status}</span>
      </div>
      <div className="progress-bar">
        <div className="progress" style={{ width: `${plan.progress * 100}%` }}></div>
      </div>
      <div className="status-meta">
        <span>Est. Cost: ${plan.estimated_cost?.toFixed(2)}</span>
        {plan.status === 'completed' && <span>Actual Cost: ${plan.actual_cost?.toFixed(2)}</span>}
      </div>
    </div>
  );
};

const ResultsDisplay = ({ plan }: { plan: Plan }) => {
  // Check for intelligent visualization plan (NEW)
  const intelligentViz = plan.context?.intelligent_visualization_planning;
  const hasIntelligentViz = intelligentViz?.status === 'completed' && intelligentViz.visualization_plan;
  
  // DEBUG LOGGING
  console.log('🔍 ResultsDisplay Debug:', {
    hasIntelligentViz,
    intelligentVizStatus: intelligentViz?.status,
    hasVisualizationPlan: !!intelligentViz?.visualization_plan,
    planKeys: intelligentViz ? Object.keys(intelligentViz) : [],
    contextKeys: plan.context ? Object.keys(plan.context) : []
  });
  
  // Fallback to legacy visualization
  const hasChart = plan.context?.visualizations?.charts?.length ? plan.context.visualizations.charts.length > 0 : false;
  const hasTable = plan.context?.query_results?.data?.length ? plan.context.query_results.data.length > 0 : false;

  // Smart default tab: prefer intelligent view if available
  const defaultTab = hasIntelligentViz ? 'intelligent' : (hasChart ? 'chart' : 'table');
  console.log('🎯 Default tab selected:', defaultTab, '| hasIntelligentViz:', hasIntelligentViz);
  const [activeTab, setActiveTab] = useState(defaultTab);

  // Get data for visualization
  const getData = () => {
    return plan.context?.query_results?.data || plan.context?.data || [];
  };

  // If no results at all
  if (!hasIntelligentViz && !hasChart && !hasTable) {
    return <div className="results-display">No results to display.</div>;
  }

  return (
    <div className="results-display">
      <div className="results-tabs">
        {hasIntelligentViz && (
          <button 
            className={activeTab === 'intelligent' ? 'active' : ''} 
            onClick={() => setActiveTab('intelligent')}
          >
            <FiBarChart2 /> Intelligent View
          </button>
        )}
        {hasChart && (
          <button 
            className={activeTab === 'chart' ? 'active' : ''} 
            onClick={() => setActiveTab('chart')}
          >
            <FiBarChart2 /> Chart
          </button>
        )}
        {hasTable && (
          <button 
            className={activeTab === 'table' ? 'active' : ''} 
            onClick={() => setActiveTab('table')}
          >
            <FiTable /> Table
          </button>
        )}
      </div>
      <div className="results-content">
        {activeTab === 'intelligent' && hasIntelligentViz && (() => {
          console.log('🎨 Rendering AdaptiveLayout with:', {
            plan: intelligentViz.visualization_plan,
            dataLength: getData().length
          });
          return (
            <AdaptiveLayout 
              plan={intelligentViz.visualization_plan!} 
              data={getData()} 
            />
          );
        })()}
        {activeTab === 'chart' && hasChart && (
          <ChartView chart={plan.context.visualizations!.charts![0]} />
        )}
        {activeTab === 'table' && hasTable && (
          <TableView data={plan.context.query_results!.data!} />
        )}
      </div>
    </div>
  );
};

const ChartView = ({ chart }: { chart: Chart }) => {
  // This is a mock chart view. In a real app, you'd use a library like Plotly or Chart.js
  return (
    <div className="chart-view">
      <h4>{chart.title}</h4>
      <div className="mock-chart">
        <p>Chart Type: {chart.type}</p>
        <p>Data points: {Array.isArray(chart.data) ? chart.data.length : 'N/A'}</p>
      </div>
    </div>
  );
};

const TableView = ({ data }: { data: any[] }) => {
  const headers = Object.keys(data[0] || {});
  return (
    <div className="table-view">
      <table>
        <thead>
          <tr>
            {headers.map(h => <th key={h}>{h}</th>)}
          </tr>
        </thead>
        <tbody>
          {data.map((row: any, i: number) => (
            <tr key={i}>
              {headers.map(h => <td key={h}>{row[h]}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const ApprovalPrompt = ({ planId }: { planId: string }) => {
  const handleApprove = async () => {
    await mockApi.approvePlan(planId, 'approver_user');
    // The polling mechanism will update the UI
  };
  return (
    <div className="approval-prompt">
      <FiAlertTriangle />
      <span>This query requires approval due to high estimated cost.</span>
      <button onClick={handleApprove}>Approve</button>
    </div>
  );
};

const PlanDetails = ({ plan }: { plan: Plan }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (plan.context.generated_sql) {
      navigator.clipboard.writeText(plan.context.generated_sql);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="plan-details">
      <h4>Reasoning</h4>
      <ul className="reasoning-steps">
        {plan.reasoning_steps.map((step: string, i: number) => <li key={i}>{step}</li>)}
      </ul>
      
      <h4>Execution Steps</h4>
      <ul className="execution-steps">
        {plan.execution_steps.map(step => (
          <li key={step.step_id} className={`status-${step.status}`}>
            {step.tool_type}
          </li>
        ))}
      </ul>

      {plan.context.generated_sql && (
        <div className="sql-display">
          <h4>Generated SQL</h4>
          <pre>
            <code>{plan.context.generated_sql}</code>
          </pre>
          <button onClick={handleCopy} className="copy-btn">
            {copied ? <FiCheck /> : <FiCopy />}
          </button>
        </div>
      )}
    </div>
  );
};

export default EnterpriseAgenticUI;
