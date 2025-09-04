import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './ConversationalQueryPanel.css';

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

interface ConversationalQueryPanelProps {
  onQueryResults?: (results: any) => void;
  onInsightUpdate?: (insight: string) => void;
}

const ConversationalQueryPanel: React.FC<ConversationalQueryPanelProps> = ({ 
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
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [showSidebar, setShowSidebar] = useState(true);
  const [sidebarTab, setSidebarTab] = useState<'history' | 'settings' | 'help'>('history');
  const [queryHistory, setQueryHistory] = useState<any[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const generateJobId = () => {
    return `azure_analytics_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  };

  const handleSendMessage = async () => {
    if (!currentInput.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: currentInput.trim(),
      timestamp: new Date()
    };

    const loadingMessage: Message = {
      id: (Date.now() + 1).toString(),
      type: 'assistant',
      content: 'Analyzing your request...',
      timestamp: new Date(),
      isLoading: true
    };

    setMessages(prev => [...prev, userMessage, loadingMessage]);
    setCurrentInput('');
    setIsLoading(true);

    try {
      const jobId = generateJobId();
      const response = await axios.post('http://localhost:8000/query', {
        natural_language: currentInput.trim(),
        job_id: jobId
      }, {
        timeout: 120000
      });

      const assistantMessage: Message = {
        id: (Date.now() + 2).toString(),
        type: 'assistant',
        content: response.data.message || 'Analysis completed successfully!',
        timestamp: new Date(),
        data: response.data,
        plotlySpec: response.data.plotly_spec,
        queryResults: response.data
      };

      // Remove loading message and add response
      setMessages(prev => prev.slice(0, -1).concat(assistantMessage));

      // Notify parent components
      if (onQueryResults) {
        onQueryResults(response.data);
      }
      if (onInsightUpdate && response.data.message) {
        onInsightUpdate(response.data.message);
      }

      // Add to query history
      const historyEntry = {
        id: Date.now().toString(),
        query: currentInput.trim(),
        response: response.data,
        timestamp: new Date(),
        success: true
      };
      setQueryHistory(prev => [historyEntry, ...prev]);

    } catch (error: any) {
      const errorMessage: Message = {
        id: (Date.now() + 3).toString(),
        type: 'assistant',
        content: `I encountered an error: ${error.response?.data?.message || error.message || 'Unknown error occurred'}. Please try rephrasing your question or check if the data source is accessible.`,
        timestamp: new Date()
      };

      // Remove loading message and add error
      setMessages(prev => prev.slice(0, -1).concat(errorMessage));
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearConversation = () => {
    setMessages([
      {
        id: '1',
        type: 'assistant',
        content: 'Conversation cleared. How can I help you with your Azure Analytics data today?',
        timestamp: new Date()
      }
    ]);
  };

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  const toggleSidebar = () => {
    setShowSidebar(prev => !prev);
  };

  const downloadData = (data: any, filename: string, format: 'csv' | 'json' | 'excel') => {
    if (!data?.rows || !data?.columns) return;

    if (format === 'csv') {
      const csvContent = [
        data.columns.join(','),
        ...data.rows.map((row: any[]) => row.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(','))
      ].join('\n');
      
      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${filename}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } else if (format === 'json') {
      const jsonContent = JSON.stringify({
        columns: data.columns,
        rows: data.rows,
        metadata: {
          timestamp: new Date().toISOString(),
          recordCount: data.rows.length
        }
      }, null, 2);
      
      const blob = new Blob([jsonContent], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${filename}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  const rerunQuery = (query: string) => {
    setCurrentInput(query);
    // Auto-focus input for user to modify or send
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  const formatTimestamp = (timestamp: Date) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const renderMessage = (message: Message) => {
    return (
      <div key={message.id} className={`message ${message.type}`}>
        <div className="message-header">
          <div className="message-avatar">
            {message.type === 'user' ? '👤' : '🤖'}
          </div>
          <div className="message-info">
            <span className="message-sender">
              {message.type === 'user' ? 'You' : 'Azure Analytics AI'}
            </span>
            <span className="message-time">
              {formatTimestamp(message.timestamp)}
            </span>
          </div>
        </div>
        
        <div className="message-content">
          {message.isLoading ? (
            <div className="loading-indicator">
              <div className="loading-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          ) : (
            <>
              <div className="message-text">{message.content}</div>
              
              {/* Render data table if available */}
              {message.queryResults?.rows && (
                <div className="data-results">
                  <div className="data-header">
                    <h4>📊 Query Results ({message.queryResults.rows.length} rows)</h4>
                    <div className="data-actions">
                      <button 
                        className="action-btn"
                        onClick={() => downloadData(message.queryResults, `query_${message.id}`, 'csv')}
                        title="Download as CSV"
                      >
                        📄 CSV
                      </button>
                      <button 
                        className="action-btn"
                        onClick={() => downloadData(message.queryResults, `query_${message.id}`, 'json')}
                        title="Download as JSON"
                      >
                        📋 JSON
                      </button>
                    </div>
                  </div>
                  <div className="data-table-container">
                    <table className="data-table">
                      <thead>
                        <tr>
                          {message.queryResults.columns?.map((col: string, idx: number) => (
                            <th key={idx}>{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {message.queryResults.rows.slice(0, 10).map((row: any[], idx: number) => (
                          <tr key={idx}>
                            {row.map((cell: any, cellIdx: number) => (
                              <td key={cellIdx}>{String(cell)}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {message.queryResults.rows.length > 10 && (
                      <div className="data-pagination">
                        Showing 10 of {message.queryResults.rows.length} rows
                        <button 
                          className="action-btn small"
                          onClick={() => {
                            // Show all data in a modal or expandable view
                            console.log('Show all data:', message.queryResults);
                          }}
                        >
                          View All
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Render chart if available */}
              {message.plotlySpec && (
                <div className="chart-container">
                  <div className="chart-header">
                    <h4>📈 Visualization</h4>
                    <div className="chart-actions">
                      <button 
                        className="action-btn"
                        onClick={() => {
                          // Download chart as PNG
                          const plotElement = document.querySelector('.js-plotly-plot');
                          if (plotElement) {
                            // @ts-ignore
                            window.Plotly?.downloadImage(plotElement, {
                              format: 'png',
                              filename: `chart_${message.id}`,
                              width: 800,
                              height: 600
                            });
                          }
                        }}
                        title="Download Chart as PNG"
                      >
                        🖼️ PNG
                      </button>
                    </div>
                  </div>
                  <Plot
                    data={message.plotlySpec.data}
                    layout={{
                      ...message.plotlySpec.layout,
                      autosize: true,
                      margin: { t: 40, r: 20, b: 40, l: 60 }
                    }}
                    config={{ 
                      responsive: true,
                      displayModeBar: true,
                      modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
                    }}
                    style={{ width: '100%', height: '400px' }}
                  />
                </div>
              )}
            </>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className={`conversational-panel ${theme}`}>
      {/* Header */}
      <div className="conversational-header">
        <div className="header-content">
          <div className="header-title">
            <span className="header-icon">☁️</span>
            <div>
              <h2>Azure Analytics Intelligence</h2>
              <p>Conversational data analysis platform</p>
            </div>
          </div>
          <div className="header-controls">
            <button 
              className="control-btn"
              onClick={toggleSidebar}
              title="Toggle sidebar"
            >
              📱
            </button>
            <button 
              className="control-btn"
              onClick={toggleTheme}
              title={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
            >
              {theme === 'light' ? '🌙' : '☀️'}
            </button>
            <button 
              className="control-btn"
              onClick={clearConversation}
              title="Clear conversation"
            >
              🗑️
            </button>
          </div>
        </div>
      </div>

      <div className="main-container">
        {/* Sidebar */}
        {showSidebar && (
          <div className="sidebar">
            <div className="sidebar-tabs">
              <button 
                className={`sidebar-tab ${sidebarTab === 'history' ? 'active' : ''}`}
                onClick={() => setSidebarTab('history')}
              >
                🕒 History
              </button>
              <button 
                className={`sidebar-tab ${sidebarTab === 'settings' ? 'active' : ''}`}
                onClick={() => setSidebarTab('settings')}
              >
                ⚙️ Settings
              </button>
              <button 
                className={`sidebar-tab ${sidebarTab === 'help' ? 'active' : ''}`}
                onClick={() => setSidebarTab('help')}
              >
                ❓ Help
              </button>
            </div>

            <div className="sidebar-content">
              {/* Query History Tab */}
              {sidebarTab === 'history' && (
                <div className="history-panel">
                  <h3>Query History</h3>
                  {queryHistory.length === 0 ? (
                    <p className="empty-state">No queries yet. Start a conversation!</p>
                  ) : (
                    <div className="history-list">
                      {queryHistory.map((item) => (
                        <div key={item.id} className="history-item">
                          <div className="history-query" title={item.query}>
                            {item.query.length > 50 ? `${item.query.substring(0, 50)}...` : item.query}
                          </div>
                          <div className="history-meta">
                            {item.timestamp.toLocaleTimeString()}
                            {item.response?.rows && (
                              <span className="history-results">
                                {item.response.rows.length} rows
                              </span>
                            )}
                          </div>
                          <div className="history-actions">
                            <button 
                              className="history-btn"
                              onClick={() => rerunQuery(item.query)}
                              title="Use this query"
                            >
                              🔄
                            </button>
                            {item.response?.rows && (
                              <button 
                                className="history-btn"
                                onClick={() => downloadData(item.response, `history_${item.id}`, 'csv')}
                                title="Download results"
                              >
                                📥
                              </button>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Settings Tab */}
              {sidebarTab === 'settings' && (
                <div className="settings-panel">
                  <h3>Settings</h3>
                  <div className="setting-group">
                    <label>Theme</label>
                    <button className="setting-btn" onClick={toggleTheme}>
                      {theme === 'light' ? '🌙 Switch to Dark' : '☀️ Switch to Light'}
                    </button>
                  </div>
                  <div className="setting-group">
                    <label>Auto-scroll to new messages</label>
                    <input type="checkbox" defaultChecked />
                  </div>
                  <div className="setting-group">
                    <label>Show timestamps</label>
                    <input type="checkbox" defaultChecked />
                  </div>
                  <div className="setting-group">
                    <label>Default export format</label>
                    <select className="setting-select">
                      <option value="csv">CSV</option>
                      <option value="json">JSON</option>
                      <option value="excel">Excel</option>
                    </select>
                  </div>
                </div>
              )}

              {/* Help Tab */}
              {sidebarTab === 'help' && (
                <div className="help-panel">
                  <h3>Help & Tips</h3>
                  <div className="help-section">
                    <h4>💡 Getting Started</h4>
                    <ul>
                      <li>Ask questions in natural language</li>
                      <li>Request specific tables or data</li>
                      <li>Ask for visualizations and charts</li>
                    </ul>
                  </div>
                  <div className="help-section">
                    <h4>📊 Example Queries</h4>
                    <div className="example-queries">
                      <button 
                        className="example-btn"
                        onClick={() => setCurrentInput('Show me the top 10 revenue sources')}
                      >
                        "Show me the top 10 revenue sources"
                      </button>
                      <button 
                        className="example-btn"
                        onClick={() => setCurrentInput('Create a chart of user engagement over time')}
                      >
                        "Create a chart of user engagement over time"
                      </button>
                      <button 
                        className="example-btn"
                        onClick={() => setCurrentInput('Analyze performance metrics by region')}
                      >
                        "Analyze performance metrics by region"
                      </button>
                    </div>
                  </div>
                  <div className="help-section">
                    <h4>🔧 Features</h4>
                    <ul>
                      <li>Download results as CSV/JSON</li>
                      <li>Save charts as PNG</li>
                      <li>Query history and rerun</li>
                      <li>Dark/Light theme switching</li>
                    </ul>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Messages Container */}
        <div className="messages-container">
          <div className="messages-list">
            {messages.map(renderMessage)}
            <div ref={messagesEndRef} />
          </div>
        </div>
      </div>

      {/* Input Container */}
      <div className="input-container">
        <div className="input-wrapper">
          <textarea
            ref={inputRef}
            value={currentInput}
            onChange={(e) => setCurrentInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me anything about your Azure Analytics data... (Press Enter to send, Shift+Enter for new line)"
            className="message-input"
            rows={1}
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={!currentInput.trim() || isLoading}
            className="send-button"
          >
            {isLoading ? (
              <div className="loading-spinner">⏳</div>
            ) : (
              '📤'
            )}
          </button>
        </div>
        
        {/* Quick Actions */}
        <div className="quick-actions">
          <button 
            className="quick-action-btn"
            onClick={() => setCurrentInput('Show me the top 10 records from my analytics data')}
            disabled={isLoading}
          >
            📊 Show Top Data
          </button>
          <button 
            className="quick-action-btn"
            onClick={() => setCurrentInput('Create a visualization of revenue trends')}
            disabled={isLoading}
          >
            📈 Revenue Analysis
          </button>
          <button 
            className="quick-action-btn"
            onClick={() => setCurrentInput('Analyze user engagement patterns')}
            disabled={isLoading}
          >
            👥 User Analytics
          </button>
          <button 
            className="quick-action-btn"
            onClick={() => setCurrentInput('Generate a performance dashboard')}
            disabled={isLoading}
          >
            📋 Performance Dashboard
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConversationalQueryPanel;
