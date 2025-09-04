import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './ModernAnalyticsChat.css';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  data?: any;
  plotlySpec?: any;
  queryResults?: any;
  isLoading?: boolean;
}

interface DatabaseConnection {
  type: 'snowflake' | 'postgresql' | 'azure-sql';
  host: string;
  database: string;
  username: string;
  password: string;
  warehouse?: string; // For Snowflake
  schema?: string;
  port?: number;
}

interface ModernAnalyticsChatProps {
  onQueryResults?: (results: any) => void;
  onInsightUpdate?: (insight: string) => void;
}

const ModernAnalyticsChat: React.FC<ModernAnalyticsChatProps> = ({ 
  onQueryResults, 
  onInsightUpdate 
}) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'assistant',
      content: 'Hello! I\'m your Azure Analytics Intelligence assistant. I can help you analyze your data using natural language. What would you like to explore today?',
      timestamp: new Date()
    }
  ]);
  
  const [currentInput, setCurrentInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'testing'>('disconnected');
  const [queryHistory, setQueryHistory] = useState<any[]>([]);
  
  const [dbConnection, setDbConnection] = useState<DatabaseConnection>({
    type: 'snowflake',
    host: '',
    database: '',
    username: '',
    password: '',
    warehouse: '',
    schema: ''
  });

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const testConnection = async () => {
    setConnectionStatus('testing');
    try {
      const response = await axios.post('/api/test-connection', dbConnection);
      if (response.data.success) {
        setConnectionStatus('connected');
      } else {
        setConnectionStatus('disconnected');
      }
    } catch (error) {
      setConnectionStatus('disconnected');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!currentInput.trim() || isLoading) return;
    
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: currentInput.trim(),
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setCurrentInput('');
    setIsLoading(true);

    const loadingMessage: Message = {
      id: (Date.now() + 1).toString(),
      type: 'assistant',
      content: 'Analyzing your request...',
      timestamp: new Date(),
      isLoading: true
    };
    
    setMessages(prev => [...prev, loadingMessage]);

    try {
      const response = await axios.post('/api/query', {
        query: userMessage.content
      });

      const assistantMessage: Message = {
        id: (Date.now() + 2).toString(),
        type: 'assistant',
        content: response.data.response || 'Here are your results:',
        timestamp: new Date(),
        data: response.data.data,
        plotlySpec: response.data.plotly_spec,
        queryResults: response.data
      };

      setMessages(prev => prev.slice(0, -1).concat(assistantMessage));
      
      // Add to history
      setQueryHistory(prev => [{
        query: userMessage.content,
        response: assistantMessage,
        timestamp: new Date()
      }, ...prev.slice(0, 9)]); // Keep last 10 queries

      if (onQueryResults) {
        onQueryResults(response.data);
      }
      
      if (onInsightUpdate && response.data.response) {
        onInsightUpdate(response.data.response);
      }
      
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 2).toString(),
        type: 'assistant',
        content: 'I apologize, but I encountered an error processing your request. Please try again or check your database connection.',
        timestamp: new Date()
      };
      setMessages(prev => prev.slice(0, -1).concat(errorMessage));
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  const downloadData = (data: any, filename: string, format: 'csv' | 'json') => {
    if (!data) return;
    
    let content = '';
    let mimeType = '';
    
    if (format === 'csv') {
      if (Array.isArray(data) && data.length > 0) {
        const headers = Object.keys(data[0]).join(',');
        const rows = data.map(row => Object.values(row).join(',')).join('\n');
        content = headers + '\n' + rows;
        mimeType = 'text/csv';
      }
    } else {
      content = JSON.stringify(data, null, 2);
      mimeType = 'application/json';
    }
    
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}.${format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const renderMessage = (message: Message) => {
    return (
      <div key={message.id} className={`message ${message.type}`}>
        <div className="message-avatar">
          {message.type === 'user' ? '👤' : '🤖'}
        </div>
        <div className="message-content">
          <div className="message-text">
            {message.isLoading ? (
              <div className="loading-dots">
                <span></span><span></span><span></span>
              </div>
            ) : (
              message.content
            )}
          </div>
          
          {message.data && !message.isLoading && (
            <div className="message-data">
              <div className="data-header">
                <span>📊 Query Results</span>
                <div className="data-actions">
                  <button 
                    onClick={() => downloadData(message.data, `query-${message.id}`, 'csv')}
                    className="action-btn"
                  >
                    CSV
                  </button>
                  <button 
                    onClick={() => downloadData(message.data, `query-${message.id}`, 'json')}
                    className="action-btn"
                  >
                    JSON
                  </button>
                </div>
              </div>
              
              {message.plotlySpec ? (
                <div className="chart-container">
                  <Plot
                    data={message.plotlySpec.data}
                    layout={{
                      ...message.plotlySpec.layout,
                      autosize: true,
                      margin: { l: 50, r: 50, t: 50, b: 50 }
                    }}
                    config={{
                      responsive: true,
                      displayModeBar: true,
                      modeBarButtonsToRemove: ['pan2d', 'lasso2d']
                    }}
                    style={{ width: '100%', height: '400px' }}
                  />
                </div>
              ) : (
                <div className="data-table">
                  <div className="table-wrapper">
                    <table>
                      <thead>
                        <tr>
                          {message.data[0] && Object.keys(message.data[0]).map(key => (
                            <th key={key}>{key}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {message.data.slice(0, 10).map((row: any, index: number) => (
                          <tr key={index}>
                            {Object.values(row).map((value: any, cellIndex: number) => (
                              <td key={cellIndex}>{String(value)}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {message.data.length > 10 && (
                      <div className="table-footer">
                        Showing 10 of {message.data.length} rows
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
          
          <div className="message-timestamp">
            {message.timestamp.toLocaleTimeString()}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="modern-analytics-chat">
      {/* Header */}
      <div className="chat-header">
        <div className="header-left">
          <div className="logo">🤖</div>
          <div className="title">
            <h1>Azure Analytics Intelligence</h1>
            <p>Natural language data exploration</p>
          </div>
        </div>
        <div className="header-right">
          <div className={`connection-status ${connectionStatus}`}>
            <div className="status-dot"></div>
            <span>{connectionStatus === 'connected' ? 'Connected' : 'Disconnected'}</span>
          </div>
          <button 
            className="header-btn" 
            onClick={() => setShowHistory(!showHistory)}
            title="Query History"
          >
            📜
          </button>
          <button 
            className="header-btn" 
            onClick={() => setShowSettings(!showSettings)}
            title="Settings"
          >
            ⚙️
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="chat-main">
        {/* Messages Area */}
        <div className="messages-container">
          <div className="messages">
            {messages.map(renderMessage)}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="input-container">
          <form onSubmit={handleSubmit} className="input-form">
            <div className="input-wrapper">
              <textarea
                ref={inputRef}
                value={currentInput}
                onChange={(e) => setCurrentInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask anything about your data..."
                className="message-input"
                rows={1}
                disabled={isLoading}
              />
              <button 
                type="submit" 
                className="send-button"
                disabled={!currentInput.trim() || isLoading}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <path d="M2 21L23 12L2 3V10L17 12L2 14V21Z" fill="currentColor"/>
                </svg>
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="settings-panel">
          <div className="panel-header">
            <h3>Database Connection</h3>
            <button onClick={() => setShowSettings(false)} className="close-btn">×</button>
          </div>
          
          <div className="settings-content">
            <div className="form-group">
              <label>Database Type</label>
              <select 
                value={dbConnection.type} 
                onChange={(e) => setDbConnection({...dbConnection, type: e.target.value as any})}
              >
                <option value="snowflake">Snowflake</option>
                <option value="postgresql">PostgreSQL</option>
                <option value="azure-sql">Azure SQL</option>
              </select>
            </div>
            
            <div className="form-group">
              <label>Host</label>
              <input 
                type="text" 
                value={dbConnection.host}
                onChange={(e) => setDbConnection({...dbConnection, host: e.target.value})}
                placeholder="your-account.snowflakecomputing.com"
              />
            </div>
            
            <div className="form-row">
              <div className="form-group">
                <label>Database</label>
                <input 
                  type="text" 
                  value={dbConnection.database}
                  onChange={(e) => setDbConnection({...dbConnection, database: e.target.value})}
                />
              </div>
              <div className="form-group">
                <label>Schema</label>
                <input 
                  type="text" 
                  value={dbConnection.schema}
                  onChange={(e) => setDbConnection({...dbConnection, schema: e.target.value})}
                />
              </div>
            </div>
            
            {dbConnection.type === 'snowflake' && (
              <div className="form-group">
                <label>Warehouse</label>
                <input 
                  type="text" 
                  value={dbConnection.warehouse}
                  onChange={(e) => setDbConnection({...dbConnection, warehouse: e.target.value})}
                />
              </div>
            )}
            
            <div className="form-row">
              <div className="form-group">
                <label>Username</label>
                <input 
                  type="text" 
                  value={dbConnection.username}
                  onChange={(e) => setDbConnection({...dbConnection, username: e.target.value})}
                />
              </div>
              <div className="form-group">
                <label>Password</label>
                <input 
                  type="password" 
                  value={dbConnection.password}
                  onChange={(e) => setDbConnection({...dbConnection, password: e.target.value})}
                />
              </div>
            </div>
            
            <div className="form-actions">
              <button 
                type="button" 
                onClick={testConnection}
                className="test-btn"
                disabled={connectionStatus === 'testing'}
              >
                {connectionStatus === 'testing' ? 'Testing...' : 'Test Connection'}
              </button>
              <button type="button" className="save-btn">
                Save Configuration
              </button>
            </div>
          </div>
        </div>
      )}

      {/* History Panel */}
      {showHistory && (
        <div className="history-panel">
          <div className="panel-header">
            <h3>Query History</h3>
            <button onClick={() => setShowHistory(false)} className="close-btn">×</button>
          </div>
          
          <div className="history-content">
            {queryHistory.length === 0 ? (
              <div className="empty-state">
                <p>No queries yet. Start asking questions about your data!</p>
              </div>
            ) : (
              queryHistory.map((item, index) => (
                <div key={index} className="history-item">
                  <div className="history-query">{item.query}</div>
                  <div className="history-time">
                    {item.timestamp.toLocaleString()}
                  </div>
                  <button 
                    className="rerun-btn"
                    onClick={() => setCurrentInput(item.query)}
                  >
                    Rerun
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModernAnalyticsChat;
