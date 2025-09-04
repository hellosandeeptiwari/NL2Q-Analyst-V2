import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './EnterpriseQueryPanel.css';

interface TableSuggestion {
  suggested_tables: string[];
  all_tables: string[];
  message: string;
  query: string;
}

interface QueryPanelProps {
  onQueryResults?: (response: any) => void;
  onInsightUpdate?: (insight: string) => void;
}

interface ErrorResponse {
  error: string;
  message?: string;
  available_columns?: string[];
  suggestions?: any;
  llm_analysis?: any;
}

function EnterpriseQueryPanel({ onQueryResults, onInsightUpdate }: QueryPanelProps = {}) {
  const [nl, setNl] = useState('');
  const [jobId, setJobId] = useState('');
  const [dbType, setDbType] = useState('snowflake');
  const [response, setResponse] = useState<any>(null);
  const [error, setError] = useState('');
  const [errorDetails, setErrorDetails] = useState<ErrorResponse | null>(null);
  const [insight, setInsight] = useState('');
  const [tableSuggestions, setTableSuggestions] = useState<TableSuggestion | null>(null);
  const [selectedTables, setSelectedTables] = useState<string[]>([]);
  const [showAllTables, setShowAllTables] = useState(false);
  const [loading, setLoading] = useState(false);
  const [insightLoading, setInsightLoading] = useState(false);
  const [queryHistory, setQueryHistory] = useState<string[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [theme, setTheme] = useState<'light' | 'dark'>('light');

  // Load query history on component mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('azure-analyst-history');
    if (savedHistory) {
      setQueryHistory(JSON.parse(savedHistory));
    }
    
    const savedTheme = localStorage.getItem('azure-analyst-theme') as 'light' | 'dark';
    if (savedTheme) {
      setTheme(savedTheme);
    }
    
    // Set default job ID if empty
    if (!jobId) {
      setJobId(`query_${Date.now()}`);
    }
  }, []);

  // Save to history
  const saveToHistory = (query: string) => {
    const newHistory = [query, ...queryHistory.filter(h => h !== query)].slice(0, 10);
    setQueryHistory(newHistory);
    localStorage.setItem('azure-analyst-history', JSON.stringify(newHistory));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!nl.trim()) return;
    
    setError('');
    setErrorDetails(null);
    setTableSuggestions(null);
    setResponse(null);
    setLoading(true);
    
    try {
      // Save to history
      saveToHistory(nl);
      
      // First, get table suggestions
      console.log('🔍 Getting table suggestions for query:', nl);
      const suggestionsRes = await axios.post('http://localhost:8000/table-suggestions', {
        query: nl,
        top_k: 5
      });

      if (suggestionsRes.data && suggestionsRes.data.suggested_tables) {
        // Handle Azure agent server response format
        const azureResponse = suggestionsRes.data;
        const tableNames = azureResponse.suggestions?.map((s: any) => s.table_name || s) || azureResponse.suggested_tables;
        
        setTableSuggestions({
          suggested_tables: tableNames,
          all_tables: azureResponse.all_tables || [],
          message: azureResponse.user_guidance?.message || 'Please select tables for your query',
          query: nl
        });
        
        console.log('✅ Table suggestions received:', tableNames);
        setSelectedTables(tableNames.slice(0, 3)); // Auto-select top 3
      }

      // Generate and execute the query
      console.log('🚀 Executing query with backend...');
      const queryRes = await axios.post('http://localhost:8000/query', {
        natural_language: nl,
        job_id: jobId || `query_${Date.now()}`,
        database_type: dbType
      });

      console.log('✅ Query response received:', queryRes.data);
      setResponse(queryRes.data);
      
      if (onQueryResults) {
        onQueryResults(queryRes.data);
      }

      // Generate insights if successful
      if (queryRes.data.success && queryRes.data.rows && queryRes.data.rows.length > 0) {
        generateInsight(queryRes.data);
      }

    } catch (err: any) {
      console.error('❌ Query failed:', err);
      
      if (err.response?.data) {
        const errorData = err.response.data;
        setErrorDetails(errorData);
        
        if (errorData.suggestions) {
          setTableSuggestions({
            suggested_tables: errorData.suggestions.suggested_tables || [],
            all_tables: errorData.suggestions.all_tables || [],
            message: errorData.suggestions.message || 'Query failed. Try these suggestions:',
            query: nl
          });
        }
        
        setError(errorData.error || errorData.message || 'Query execution failed');
      } else {
        setError('Network error. Please check if the server is running.');
      }
    } finally {
      setLoading(false);
    }
  };

  const generateInsight = async (queryResponse: any) => {
    if (!queryResponse.rows || queryResponse.rows.length === 0) return;
    
    setInsightLoading(true);
    try {
      const insightRes = await axios.post('http://localhost:8000/insight', {
        query: nl,
        results: queryResponse.rows.slice(0, 100), // Send first 100 rows for analysis
        columns: queryResponse.columns,
        job_id: jobId
      });
      
      if (insightRes.data.insight) {
        setInsight(insightRes.data.insight);
        if (onInsightUpdate) {
          onInsightUpdate(insightRes.data.insight);
        }
      }
    } catch (err) {
      console.log('Insight generation failed (optional feature)');
    } finally {
      setInsightLoading(false);
    }
  };

  const handleExampleQuery = (query: string) => {
    setNl(query);
  };

  const handleHistorySelect = (query: string) => {
    setNl(query);
  };

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    localStorage.setItem('azure-analyst-theme', newTheme);
  };

  const renderVisualization = () => {
    if (!response?.visualization) return null;

    const viz = response.visualization;
    return (
      <div className="enterprise-visualization">
        <h3 className="viz-title">📊 {viz.title || 'Data Visualization'}</h3>
        <div className="viz-container">
          <Plot
            data={viz.data}
            layout={{
              ...viz.layout,
              paper_bgcolor: theme === 'dark' ? '#1a1a1a' : '#ffffff',
              plot_bgcolor: theme === 'dark' ? '#2d2d2d' : '#f8f9fa',
              font: { 
                color: theme === 'dark' ? '#ffffff' : '#333333',
                family: 'Inter, system-ui, sans-serif'
              },
              title: {
                ...viz.layout?.title,
                font: { 
                  size: 18, 
                  color: theme === 'dark' ? '#ffffff' : '#333333' 
                }
              }
            }}
            config={{
              displayModeBar: true,
              displaylogo: false,
              modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
              responsive: true
            }}
            style={{ width: '100%', height: '500px' }}
          />
        </div>
        {viz.description && (
          <p className="viz-description">{viz.description}</p>
        )}
      </div>
    );
  };

  const renderDataGrid = () => {
    if (!response?.rows || response.rows.length === 0) return null;

    return (
      <div className="enterprise-data-grid">
        <div className="grid-header">
          <h3>📊 Azure Analytics Results ({response.rows.length} records)</h3>
          <div className="grid-actions">
            <button 
              className="btn btn-secondary"
              onClick={() => {
                const csv = [
                  response.columns.join(','),
                  ...response.rows.map((row: any[]) => row.join(','))
                ].join('\\n');
                const blob = new Blob([csv], { type: 'text/csv' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `azure-analytics-${Date.now()}.csv`;
                a.click();
              }}
            >
              📥 Export CSV
            </button>
          </div>
        </div>
        
        <div className="grid-container">
          <table className="data-table">
            <thead>
              <tr>
                {response.columns.map((col: string, idx: number) => (
                  <th key={idx} className="table-header">{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {response.rows.slice(0, 50).map((row: any[], idx: number) => (
                <tr key={idx} className="table-row">
                  {row.map((cell: any, cellIdx: number) => (
                    <td key={cellIdx} className="table-cell">
                      {cell !== null && cell !== undefined ? String(cell) : '—'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          
          {response.rows.length > 50 && (
            <div className="grid-footer">
              <p>Showing first 50 of {response.rows.length} records</p>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className={`enterprise-query-panel ${theme}`}>
      <div className="panel-header">
        <div className="header-content">
          <div className="brand">
            <h1 className="brand-title">
              ☁️ Azure Analytics Intelligence
            </h1>
            <p className="brand-subtitle">
              Enterprise-grade natural language data analysis platform
            </p>
          </div>
          
          <div className="header-actions">
            <button 
              className="btn btn-icon"
              onClick={toggleTheme}
              title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
            >
              {theme === 'light' ? '🌙' : '☀️'}
            </button>
            
            <button 
              className="btn btn-icon"
              onClick={() => setShowAdvanced(!showAdvanced)}
              title="Advanced settings"
            >
              ⚙️
            </button>
          </div>
        </div>
      </div>

      <div className="panel-body">
        <div className="query-section">
          <form onSubmit={handleSubmit} className="query-form">
            <div className="form-group">
              <label htmlFor="query-input" className="form-label">
                Natural Language Query
              </label>
              <textarea
                id="query-input"
                value={nl}
                onChange={(e) => setNl(e.target.value)}
                className="query-input"
                rows={3}
                placeholder="Example queries:&#10;• 'show me analytics data with top 10 revenue sources'&#10;• 'create a visualization of user engagement metrics'&#10;• 'analyze performance trends by region and time period'"
                disabled={loading}
              />
            </div>
            
            {showAdvanced && (
              <div className="advanced-settings">
                <div className="form-row">
                  <div className="form-group">
                    <label className="form-label">Job ID</label>
                    <input
                      type="text"
                      value={jobId}
                      onChange={(e) => setJobId(e.target.value)}
                      className="form-input"
                      placeholder="Auto-generated"
                    />
                  </div>
                  
                  <div className="form-group">
                    <label className="form-label">Database Type</label>
                    <select
                      value={dbType}
                      onChange={(e) => setDbType(e.target.value)}
                      className="form-select"
                    >
                      <option value="snowflake">Snowflake</option>
                      <option value="postgresql">PostgreSQL</option>
                      <option value="mysql">MySQL</option>
                    </select>
                  </div>
                </div>
              </div>
            )}
            
            <div className="form-actions">
              <button
                type="submit"
                disabled={loading || !nl.trim()}
                className="btn btn-primary btn-large"
              >
                {loading ? '🔄 Analyzing...' : '🚀 Analyze Data'}
              </button>
            </div>
          </form>
          
          {/* Quick Actions */}
          <div className="quick-actions">
            <h4>Quick Actions</h4>
            <div className="action-buttons">
              <button 
                className="btn btn-secondary btn-small"
                onClick={() => handleExampleQuery('show me top 10 revenue sources with trends')}
              >
                📈 Revenue Analysis
              </button>
              <button 
                className="btn btn-secondary btn-small"
                onClick={() => handleExampleQuery('analyze user engagement metrics by channel')}
              >
                👥 User Analytics
              </button>
              <button 
                className="btn btn-secondary btn-small"
                onClick={() => handleExampleQuery('create performance dashboard for last quarter')}
              >
                📊 Performance Dashboard
              </button>
            </div>
          </div>
          
          {/* Query History */}
          {queryHistory.length > 0 && (
            <div className="query-history">
              <h4>Recent Queries</h4>
              <div className="history-list">
                {queryHistory.slice(0, 5).map((query, idx) => (
                  <button
                    key={idx}
                    className="history-item"
                    onClick={() => handleHistorySelect(query)}
                  >
                    {query.length > 60 ? `${query.substring(0, 60)}...` : query}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Results Section */}
        <div className="results-section">
          {loading && (
            <div className="loading-state">
              <div className="loading-spinner"></div>
              <p>Analyzing your query with Azure Intelligence...</p>
            </div>
          )}

          {error && (
            <div className="error-state">
              <div className="error-icon">⚠️</div>
              <div className="error-content">
                <h3>Query Error</h3>
                <p>{error}</p>
                
                {errorDetails?.suggestions && (
                  <div className="error-suggestions">
                    <h4>Suggested Tables:</h4>
                    <ul>
                      {errorDetails.suggestions.suggested_tables?.map((table: string, idx: number) => (
                        <li key={idx}>{table}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}

          {tableSuggestions && (
            <div className="table-suggestions">
              <h3>📋 Table Suggestions</h3>
              <p className="suggestions-message">{tableSuggestions.message}</p>
              
              <div className="suggested-tables">
                <h4>Recommended Tables:</h4>
                <div className="table-chips">
                  {tableSuggestions.suggested_tables.map((table, idx) => (
                    <span key={idx} className="table-chip suggested">
                      {table}
                    </span>
                  ))}
                </div>
              </div>
              
              {showAllTables && tableSuggestions.all_tables.length > 0 && (
                <div className="all-tables">
                  <h4>All Available Tables:</h4>
                  <div className="table-chips">
                    {tableSuggestions.all_tables.map((table, idx) => (
                      <span key={idx} className="table-chip">
                        {table}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              
              <button
                className="btn btn-link"
                onClick={() => setShowAllTables(!showAllTables)}
              >
                {showAllTables ? 'Hide' : 'Show'} All Tables ({tableSuggestions.all_tables.length})
              </button>
            </div>
          )}

          {response?.success && (
            <div className="success-results">
              {/* Visualization */}
              {renderVisualization()}
              
              {/* Data Grid */}
              {renderDataGrid()}
              
              {/* Insights */}
              {(insight || insightLoading) && (
                <div className="insights-section">
                  <h3>🧠 AI Insights</h3>
                  {insightLoading ? (
                    <div className="insight-loading">
                      <div className="loading-spinner small"></div>
                      <span>Generating insights...</span>
                    </div>
                  ) : (
                    <div className="insight-content">
                      {insight.split('\\n').map((line, idx) => (
                        <p key={idx}>{line}</p>
                      ))}
                    </div>
                  )}
                </div>
              )}
              
              {/* Query Metadata */}
              <div className="query-metadata">
                <div className="metadata-grid">
                  <div className="metadata-item">
                    <span className="metadata-label">Query ID</span>
                    <span className="metadata-value">{response.job_id || jobId}</span>
                  </div>
                  <div className="metadata-item">
                    <span className="metadata-label">Records</span>
                    <span className="metadata-value">{response.rows?.length || 0}</span>
                  </div>
                  <div className="metadata-item">
                    <span className="metadata-label">Columns</span>
                    <span className="metadata-value">{response.columns?.length || 0}</span>
                  </div>
                  <div className="metadata-item">
                    <span className="metadata-label">Timestamp</span>
                    <span className="metadata-value">{new Date().toLocaleString()}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default EnterpriseQueryPanel;
