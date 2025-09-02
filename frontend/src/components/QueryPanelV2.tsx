import React, { useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';

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

function QueryPanel({ onQueryResults, onInsightUpdate }: QueryPanelProps = {}) {
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setErrorDetails(null);
    setTableSuggestions(null);
    setResponse(null);
    setLoading(true);
    
    try {
      // First, get table suggestions
      console.log('🔍 Getting table suggestions for query:', nl);
      const suggestionsRes = await axios.post('http://localhost:8000/table-suggestions', {
        query: nl
      });
      
      console.log('📋 Table suggestions response:', suggestionsRes.data);
      
      // Handle NBA agent server response format
      if (suggestionsRes.data.user_guidance?.should_provide_suggestions && suggestionsRes.data.suggestions?.length > 0) {
        // Convert NBA agent format to frontend format
        const nbaResponse = suggestionsRes.data;
        const tableNames = nbaResponse.suggestions.map((s: any) => s.table_name || s);
        
        const convertedSuggestions = {
          suggested_tables: tableNames,
          all_tables: tableNames, // For now, use the same list
          message: nbaResponse.user_guidance.message || 'Please select tables for your query',
          query: nl
        };
        
        console.log('🔄 Converted suggestions:', convertedSuggestions);
        setTableSuggestions(convertedSuggestions);
        setLoading(false);
        return;
      }
      
      // Handle standard backend format  
      if (suggestionsRes.data.suggested_tables?.length > 0) {
        setTableSuggestions(suggestionsRes.data);
        setLoading(false);
        return;
      }
      
      // If no suggestions needed or no matches, run query directly
      const res = await axios.post('http://localhost:8000/query', {
        natural_language: nl,
        job_id: jobId || `job_${Date.now()}`,
        db_type: dbType,
        selected_tables: selectedTables
      });
      
      setResponse(res.data);
      setSelectedTables([]); // Reset selections
      // Notify parent component of results
      if (onQueryResults) {
        onQueryResults(res.data);
      }
    } catch (err: any) {
      const errorData = err.response?.data;
      setError(errorData?.error || 'Error running query');
      console.error('❌ Query error:', errorData);
      if (errorData && (errorData.message || errorData.available_columns || errorData.llm_analysis)) {
        setErrorDetails(errorData);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleTableSelection = (tableName: string) => {
    setSelectedTables(prev => {
      if (prev.includes(tableName)) {
        return prev.filter(t => t !== tableName);
      } else {
        return [...prev, tableName];
      }
    });
  };

  const executeWithSelectedTables = async () => {
    if (selectedTables.length === 0) {
      setError('Please select at least one table');
      return;
    }

    setError('');
    setErrorDetails(null);
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/query-with-table', {
        natural_language: tableSuggestions?.query || nl,
        job_id: jobId || `job_${Date.now()}`,
        selected_tables: selectedTables
      });
      setResponse(res.data);
      setTableSuggestions(null);
      // Notify parent component of results
      if (onQueryResults) {
        onQueryResults(res.data);
      }
    } catch (err: any) {
      const errorData = err.response?.data;
      setError(errorData?.error || 'Error running query with selected tables');
      if (errorData && (errorData.message || errorData.available_columns || errorData.llm_analysis)) {
        setErrorDetails(errorData);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleInsight = async () => {
    if (!response) return;
    setInsightLoading(true);
    try {
      const insightPayload = {
        location: response.location || '',
        query: tableSuggestions?.query || nl,
        table_name: response.table_used || '',
        data_rows: response.rows || [],
        columns: response.columns || []
      };
      
      const res = await axios.post('http://localhost:8000/insights', insightPayload);
      setInsight(res.data.insight);
      // Notify parent component of insight update
      if (onInsightUpdate) {
        onInsightUpdate(res.data.insight);
      }
    } catch (err: any) {
      setError('Error generating insights: ' + (err.response?.data?.error || err.message));
    } finally {
      setInsightLoading(false);
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '2rem' }}>
        <div style={{
          width: '40px',
          height: '40px',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          borderRadius: '10px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginRight: '1rem',
          boxShadow: '0 4px 12px rgba(102, 126, 234, 0.3)'
        }}>
          <span style={{ color: 'white', fontSize: '20px' }}>🔍</span>
        </div>
        <div>
          <h2 className="gradient-text" style={{ fontSize: '1.5rem', marginBottom: '0.25rem' }}>
            Natural Language Query
          </h2>
          <p style={{ color: '#64748b', margin: 0, fontSize: '0.9rem' }}>
            Ask questions about your data in plain English
          </p>
        </div>
      </div>
      
      {/* Enhanced Main Query Form */}
      <div style={{ 
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        borderRadius: '16px',
        padding: '2rem',
        marginBottom: '2rem',
        boxShadow: '0 10px 25px rgba(0,0,0,0.1)'
      }}>
        <div style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ 
            color: 'white', 
            margin: 0, 
            fontSize: '1.5rem',
            fontWeight: '700',
            textAlign: 'center'
          }}>
            🏀 NBA Data Intelligence System
          </h2>
          <p style={{ 
            color: 'rgba(255,255,255,0.9)', 
            margin: '0.5rem 0 0 0', 
            textAlign: 'center',
            fontSize: '14px'
          }}>
            Ask questions about NBA data in natural language and get intelligent insights
          </p>
        </div>
        
        <form onSubmit={handleSubmit}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 200px 150px', gap: '1rem', alignItems: 'end', marginBottom: '1.5rem' }}>
            <div>
              <label style={{ 
                display: 'block', 
                marginBottom: '0.75rem', 
                fontWeight: '600', 
                color: 'white',
                fontSize: '0.9rem',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                🎯 Your Query
              </label>
              <textarea
                value={nl}
                onChange={e => setNl(e.target.value)}
                placeholder="Example queries:&#10;• 'read table final nba output python and fetch top 5 rows'&#10;• 'create a visualization with frequency of recommended message'&#10;• 'show me NBA player performance data with charts'"
                style={{ 
                  width: '100%', 
                  padding: '16px 20px', 
                  border: 'none', 
                  borderRadius: '12px',
                  fontSize: '14px',
                  minHeight: '120px',
                  resize: 'vertical',
                  fontFamily: 'inherit',
                  background: 'white',
                  boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
                  transition: 'all 0.2s ease',
                  outline: 'none'
                }}
                onFocus={(e) => e.target.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)'}
                onBlur={(e) => e.target.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)'}
                required
              />
            </div>
            
            <div>
              <label style={{ 
                display: 'block', 
                marginBottom: '0.75rem', 
                fontWeight: '600', 
                color: 'white',
                fontSize: '0.9rem',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                🗄️ Database
              </label>
              <select
                value={dbType}
                onChange={e => setDbType(e.target.value)}
              style={{ 
                width: '100%', 
                padding: '16px 20px', 
                border: 'none', 
                borderRadius: '12px',
                background: 'white',
                fontSize: '14px',
                boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
                outline: 'none'
              }}
            >
              <option value="snowflake">❄️ Snowflake</option>
              <option value="sqlite">💾 SQLite</option>
              <option value="postgres">🐘 PostgreSQL</option>
              <option value="azure_sql">☁️ Azure SQL</option>
            </select>
          </div>
          
          <div>
            <button
              type="submit"
              disabled={loading}
              style={{ 
                width: '100%',
                padding: '16px 24px', 
                backgroundColor: loading ? '#6c757d' : '#28a745',
                color: 'white',
                border: 'none',
                borderRadius: '12px',
                fontSize: '14px',
                fontWeight: '600',
                cursor: loading ? 'not-allowed' : 'pointer',
                boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
                transition: 'all 0.2s ease',
                textTransform: 'uppercase',
                letterSpacing: '0.5px'
              }}
              onMouseEnter={(e) => {
                if (!loading) {
                  e.currentTarget.style.backgroundColor = '#218838';
                  e.currentTarget.style.transform = 'translateY(-1px)';
                  e.currentTarget.style.boxShadow = '0 6px 12px rgba(0,0,0,0.15)';
                }
              }}
              onMouseLeave={(e) => {
                if (!loading) {
                  e.currentTarget.style.backgroundColor = '#28a745';
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
                }
              }}
            >
              {loading ? '🔄 Analyzing...' : '🚀 Query NBA Data'}
            </button>
          </div>
        </div>
        </form>
      </div>

      {/* Table Selection UI */}
      {tableSuggestions && (
        <div style={{ 
          marginTop: '2rem', 
          padding: '2rem', 
          background: 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)', 
          border: '2px solid #e2e8f0',
          borderRadius: '16px',
          boxShadow: '0 4px 25px rgba(0, 0, 0, 0.05)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1.5rem' }}>
            <div style={{
              width: '32px',
              height: '32px',
              background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginRight: '1rem'
            }}>
              <span style={{ color: 'white', fontSize: '16px' }}>🔍</span>
            </div>
            <div>
              <h3 style={{ color: '#374151', marginBottom: '0.25rem', fontSize: '1.2rem', fontWeight: '600' }}>
                Table Selection Required
              </h3>
              <p style={{ color: '#6b7280', margin: 0, fontSize: '0.9rem' }}>
                {tableSuggestions?.message || 'Select tables to proceed with your query'}
              </p>
            </div>
          </div>
          
          {/* Suggested Tables */}
          {tableSuggestions && tableSuggestions.suggested_tables && tableSuggestions.suggested_tables.length > 0 && (
            <div style={{ marginBottom: '2rem' }}>
              <h4 style={{ 
                color: '#374151', 
                marginBottom: '1rem', 
                fontSize: '1rem',
                fontWeight: '600',
                display: 'flex',
                alignItems: 'center'
              }}>
                <span style={{ marginRight: '0.5rem' }}>💡</span>
                AI Recommended Tables
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
                {tableSuggestions?.suggested_tables?.map((table, index) => (
                  <button
                    key={index}
                    onClick={() => handleTableSelection(table)}
                    className="table-suggestion"
                    style={{
                      ...(selectedTables.includes(table) ? { 
                        background: 'linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)',
                        borderColor: '#3b82f6',
                        color: '#1e40af'
                      } : {})
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <span style={{ fontWeight: '600', fontSize: '14px' }}>
                        {selectedTables.includes(table) ? '✓ ' : ''}
                        {table}
                      </span>
                      <span style={{
                        padding: '4px 8px',
                        background: selectedTables.includes(table) ? '#3b82f6' : '#e5e7eb',
                        color: selectedTables.includes(table) ? 'white' : '#6b7280',
                        borderRadius: '12px',
                        fontSize: '10px',
                        fontWeight: '600',
                        textTransform: 'uppercase'
                      }}>
                        {selectedTables.includes(table) ? 'Selected' : 'Select'}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* All Tables Toggle */}
          <div style={{ marginBottom: '1rem' }}>
            <button
              onClick={() => setShowAllTables(!showAllTables)}
              style={{
                padding: '6px 12px',
                border: '1px solid #6c757d',
                backgroundColor: 'transparent',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '14px',
                color: '#6c757d'
              }}
            >
              {showAllTables ? 'Hide' : 'Show'} All Available Tables ({tableSuggestions?.all_tables?.length || 0})
            </button>
          </div>

          {/* All Tables List */}
          {showAllTables && (
            <div style={{ marginBottom: '1.5rem' }}>
              <h4 style={{ color: '#495057', marginBottom: '0.5rem', fontSize: '1rem' }}>
                📋 All Available Tables:
              </h4>
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', 
                gap: '0.5rem',
                maxHeight: '200px',
                overflowY: 'auto',
                padding: '0.5rem',
                border: '1px solid #ddd',
                borderRadius: '4px',
                backgroundColor: 'white'
              }}>
                {tableSuggestions?.all_tables?.map((table, index) => (
                  <button
                    key={index}
                    onClick={() => handleTableSelection(table)}
                    style={{
                      padding: '6px 12px',
                      border: selectedTables.includes(table) ? '2px solid #007bff' : '1px solid #e9ecef',
                      backgroundColor: selectedTables.includes(table) ? '#e7f3ff' : 'white',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '13px',
                      textAlign: 'left',
                      fontWeight: selectedTables.includes(table) ? '600' : '400'
                    }}
                  >
                    {selectedTables.includes(table) ? '✓ ' : ''}{table}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Selected Tables Display */}
          {selectedTables.length > 0 && (
            <div style={{ marginBottom: '2rem' }}>
              <h4 style={{ 
                color: '#374151', 
                marginBottom: '1rem', 
                fontSize: '1rem',
                fontWeight: '600',
                display: 'flex',
                alignItems: 'center'
              }}>
                <span style={{ marginRight: '0.5rem' }}>✅</span>
                Selected Tables ({selectedTables.length})
              </h4>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem' }}>
                {selectedTables.map((table, index) => (
                  <span
                    key={index}
                    style={{
                      padding: '8px 16px',
                      background: 'linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)',
                      border: '2px solid #10b981',
                      borderRadius: '20px',
                      fontSize: '14px',
                      color: '#065f46',
                      fontWeight: '600',
                      display: 'flex',
                      alignItems: 'center'
                    }}
                  >
                    <span style={{ marginRight: '0.5rem' }}>📋</span>
                    {table}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Execute Button */}
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <button
              onClick={executeWithSelectedTables}
              disabled={selectedTables.length === 0 || loading}
              className={selectedTables.length > 0 && !loading ? 'success-button' : ''}
              style={{
                padding: '16px 32px',
                fontSize: '16px',
                fontWeight: '600',
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                background: (selectedTables.length === 0 || loading) ? '#d1d5db' : undefined,
                cursor: (selectedTables.length === 0 || loading) ? 'not-allowed' : 'pointer',
                color: (selectedTables.length === 0 || loading) ? '#6b7280' : 'white'
              }}
            >
              {loading && (
                <div className="loading-spinner" style={{ width: '18px', height: '18px' }} />
              )}
              <span style={{ fontSize: '18px' }}>
                {loading ? '⏳' : selectedTables.length > 0 ? '🚀' : '⚠️'}
              </span>
              {loading ? 'Processing Query...' : 
               selectedTables.length > 0 ? 'Execute Query with Selected Tables' : 'Select Tables First'}
            </button>
          </div>
        </div>
      )}

      {/* Enhanced Error Display */}
      {error && (
        <div className="error-alert" style={{ 
          display: 'flex', 
          alignItems: 'flex-start',
          gap: '1rem',
          flexDirection: 'column'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{
              width: '32px',
              height: '32px',
              background: '#f56565',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0
            }}>
              <span style={{ color: 'white', fontSize: '16px' }}>❌</span>
            </div>
            <div>
              <h4 style={{ margin: 0, marginBottom: '0.25rem', fontWeight: '600' }}>
                Query Error
              </h4>
              <p style={{ margin: 0, fontSize: '14px', opacity: 0.9 }}>
                {error}
              </p>
            </div>
          </div>
          
          {/* Enhanced Error Details */}
          {errorDetails && (
            <div style={{ width: '100%', marginTop: '1rem' }}>
              {errorDetails.message && (
                <div style={{ 
                  background: '#fff3cd', 
                  border: '1px solid #ffeaa7', 
                  padding: '1rem', 
                  borderRadius: '8px',
                  marginBottom: '1rem'
                }}>
                  <h5 style={{ margin: '0 0 0.5rem 0', color: '#856404' }}>💡 Suggestions</h5>
                  <pre style={{ 
                    whiteSpace: 'pre-wrap', 
                    fontSize: '13px', 
                    color: '#856404',
                    margin: 0,
                    fontFamily: 'inherit'
                  }}>
                    {errorDetails.message}
                  </pre>
                </div>
              )}
              
              {errorDetails.llm_analysis && (
                <div style={{ 
                  background: '#e7f3ff', 
                  border: '1px solid #b8daff', 
                  padding: '1rem', 
                  borderRadius: '8px'
                }}>
                  <h5 style={{ margin: '0 0 0.5rem 0', color: '#004085' }}>🤖 AI Analysis</h5>
                  
                  {errorDetails.llm_analysis.column_matches?.length > 0 && (
                    <div style={{ marginBottom: '1rem' }}>
                      <strong style={{ color: '#004085' }}>Possible Column Matches:</strong>
                      {errorDetails.llm_analysis.column_matches?.map((match: any, index: number) => (
                        <div key={index} style={{ 
                          margin: '0.5rem 0', 
                          padding: '0.5rem', 
                          background: 'rgba(255,255,255,0.7)', 
                          borderRadius: '4px' 
                        }}>
                          <div style={{ fontWeight: '600' }}>
                            '{match.requested}' → '{match.matched}' ({Math.round(match.confidence * 100)}% match)
                          </div>
                          <div style={{ fontSize: '12px', color: '#6c757d' }}>
                            {match.reason}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  
                  {errorDetails.llm_analysis.suggested_alternatives?.length > 0 && (
                    <div>
                      <strong style={{ color: '#004085' }}>Alternative Columns:</strong>
                      {errorDetails.llm_analysis.suggested_alternatives?.slice(0, 3).map((alt: any, index: number) => (
                        <div key={index} style={{ 
                          margin: '0.5rem 0', 
                          padding: '0.5rem', 
                          background: 'rgba(255,255,255,0.7)', 
                          borderRadius: '4px' 
                        }}>
                          <div style={{ fontWeight: '600' }}>'{alt.column}'</div>
                          <div style={{ fontSize: '12px', color: '#6c757d' }}>
                            {alt.relevance}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
              
              {errorDetails.available_columns && (
                <div style={{ 
                  background: '#f8f9fa', 
                  border: '1px solid #dee2e6', 
                  padding: '1rem', 
                  borderRadius: '8px',
                  marginTop: '1rem'
                }}>
                  <h5 style={{ margin: '0 0 0.5rem 0', color: '#495057' }}>📋 Available Columns</h5>
                  <div style={{ 
                    display: 'grid', 
                    gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', 
                    gap: '0.5rem',
                    fontSize: '13px'
                  }}>
                    {errorDetails.available_columns?.slice(0, 20).map((col, index) => (
                      <div key={index} style={{ 
                        padding: '0.25rem 0.5rem', 
                        background: 'white', 
                        borderRadius: '4px',
                        border: '1px solid #e9ecef'
                      }}>
                        {col}
                      </div>
                    ))}
                    {errorDetails.available_columns && errorDetails.available_columns.length > 20 && (
                      <div style={{ padding: '0.25rem 0.5rem', color: '#6c757d' }}>
                        ... and {errorDetails.available_columns.length - 20} more
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Response Display */}
      {response && (
        <div style={{ marginTop: '2rem' }}>
          <h3 style={{ color: '#495057', marginBottom: '1rem' }}>Query Results</h3>
          
          {/* Status and Info */}
          <div style={{ 
            marginBottom: '1rem', 
            padding: '1rem', 
            background: '#d4edda', 
            border: '1px solid #c3e6cb',
            borderRadius: '4px' 
          }}>
            <p><strong>Status:</strong> {response.status}</p>
            <p><strong>Message:</strong> {response.message}</p>
            {response.table_used && <p><strong>Table Used:</strong> {response.table_used}</p>}
            {response.job_id && <p><strong>Job ID:</strong> {response.job_id}</p>}
          </div>

          {/* Enhanced Data Table with Better Formatting */}
          {response.rows && response.rows.length > 0 && (
            <div style={{ marginBottom: '2rem' }}>
              <h4 style={{ color: '#495057', marginBottom: '1rem' }}>
                📊 NBA Data Results ({response.rows.length} records)
              </h4>
              
              {/* Executive Summary if available */}
              {response.executive_summary && (
                <div style={{ 
                  backgroundColor: '#e8f4f8', 
                  padding: '1rem', 
                  borderRadius: '8px', 
                  marginBottom: '1rem',
                  border: '1px solid #bee5eb'
                }}>
                  <h5 style={{ color: '#0c5460', marginBottom: '0.5rem' }}>📋 Executive Summary</h5>
                  <p style={{ margin: '0.25rem 0', fontSize: '14px' }}>
                    <strong>Query Intent:</strong> {response.executive_summary.query_intent}
                  </p>
                  <p style={{ margin: '0.25rem 0', fontSize: '14px' }}>
                    <strong>Data Source:</strong> {response.executive_summary.data_source}
                  </p>
                  {response.executive_summary.key_findings && response.executive_summary.key_findings.length > 0 && (
                    <div style={{ marginTop: '0.5rem' }}>
                      <strong style={{ fontSize: '14px' }}>Key Findings:</strong>
                      <ul style={{ margin: '0.25rem 0', paddingLeft: '1.5rem' }}>
                        {response.executive_summary.key_findings.map((finding: string, idx: number) => (
                          <li key={idx} style={{ fontSize: '14px', margin: '0.25rem 0' }}>{finding}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
              
              <div style={{ 
                maxHeight: '500px', 
                overflow: 'auto', 
                border: '1px solid #dee2e6',
                borderRadius: '8px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
                  {/* Enhanced Table Header */}
                  {response.columns && response.columns.length > 0 && (
                    <thead style={{ backgroundColor: '#343a40', color: 'white', position: 'sticky', top: 0 }}>
                      <tr>
                        <th style={{ padding: '12px 8px', textAlign: 'left', fontWeight: '600', fontSize: '13px' }}>
                          #
                        </th>
                        {response.columns.map((column: string, idx: number) => (
                          <th key={idx} style={{ 
                            padding: '12px 8px', 
                            textAlign: 'left', 
                            fontWeight: '600',
                            fontSize: '13px',
                            borderLeft: idx > 0 ? '1px solid #495057' : 'none'
                          }}>
                            {column}
                          </th>
                        ))}
                      </tr>
                    </thead>
                  )}
                  <tbody>
                    {response.rows.slice(0, 15).map((row: any, index: number) => (
                      <tr key={index} style={{ 
                        borderBottom: '1px solid #e9ecef',
                        backgroundColor: index % 2 === 0 ? '#f8f9fa' : 'white',
                        transition: 'background-color 0.2s'
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#e3f2fd'}
                      onMouseLeave={(e) => e.currentTarget.style.backgroundColor = index % 2 === 0 ? '#f8f9fa' : 'white'}>
                        <td style={{ 
                          padding: '10px 8px', 
                          textAlign: 'center',
                          fontWeight: '500',
                          color: '#6c757d',
                          fontSize: '12px'
                        }}>
                          {index + 1}
                        </td>
                        {typeof row === 'object' && !Array.isArray(row) ? (
                          // Handle formatted object rows from backend
                          Object.values(row).map((cell: any, cellIndex: number) => (
                            <td key={cellIndex} style={{ 
                              padding: '10px 8px', 
                              textAlign: 'left',
                              fontSize: '13px',
                              fontFamily: typeof cell === 'number' ? 'monospace' : 'inherit',
                              color: typeof cell === 'number' ? '#0d6efd' : '#212529'
                            }}>
                              {String(cell)}
                            </td>
                          ))
                        ) : (
                          // Handle array rows (fallback)
                          (Array.isArray(row) ? row : Object.values(row)).map((cell: any, cellIndex: number) => (
                            <td key={cellIndex} style={{ 
                              padding: '10px 8px', 
                              textAlign: 'left',
                              fontSize: '13px',
                              fontFamily: typeof cell === 'number' ? 'monospace' : 'inherit',
                              color: typeof cell === 'number' ? '#0d6efd' : '#212529'
                            }}>
                              {typeof cell === 'number' ? 
                                (cell < 1 && cell > 0 ? `${(cell * 100).toFixed(2)}%` : 
                                 cell > 1000 ? cell.toLocaleString() : 
                                 cell.toFixed(3)) : 
                                String(cell)}
                            </td>
                          ))
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {response.rows.length > 15 && (
                <p style={{ 
                  fontSize: '14px', 
                  color: '#6c757d', 
                  marginTop: '0.75rem',
                  textAlign: 'center',
                  fontStyle: 'italic'
                }}>
                  📄 Showing first 15 of {response.rows.length} records
                </p>
              )}
              
              {/* Performance Metrics */}
              {response.performance_metrics && (
                <div style={{ 
                  marginTop: '1rem', 
                  padding: '0.75rem', 
                  backgroundColor: '#f8f9fa', 
                  borderRadius: '6px',
                  fontSize: '12px',
                  color: '#495057'
                }}>
                  ⚡ Query executed in {response.performance_metrics.total_execution_time}s 
                  ({response.performance_metrics.rows_per_second} rows/sec) • 
                  Performance: {response.performance_metrics.performance_category}
                </div>
              )}
            </div>
          )}

          {/* Enhanced Visualization */}
          {response.plotly_spec && (
            <div style={{ marginBottom: '2rem' }}>
              <h4 style={{ color: '#495057', marginBottom: '1rem' }}>
                📈 Data Visualization
              </h4>
              
              {/* Visualization Context */}
              {response.analysis_insights && response.analysis_insights.visualization_type && (
                <div style={{ 
                  backgroundColor: '#fff3cd', 
                  border: '1px solid #ffeaa7',
                  borderRadius: '6px',
                  padding: '0.75rem',
                  marginBottom: '1rem'
                }}>
                  <p style={{ margin: 0, fontSize: '14px', color: '#856404' }}>
                    <strong>📊 Chart Type:</strong> {response.analysis_insights.visualization_type.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
                    {response.analysis_insights.confidence_score && (
                      <span style={{ marginLeft: '1rem' }}>
                        <strong>🎯 AI Confidence:</strong> {(response.analysis_insights.confidence_score * 100).toFixed(0)}%
                      </span>
                    )}
                  </p>
                </div>
              )}
              
              <div style={{ 
                border: '1px solid #dee2e6',
                borderRadius: '8px',
                padding: '1rem',
                backgroundColor: 'white',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
              }}>
                <Plot
                  data={response.plotly_spec.data}
                  layout={{
                    ...response.plotly_spec.layout,
                    font: { family: 'Inter, system-ui, sans-serif', size: 12 },
                    margin: { l: 60, r: 40, t: 80, b: 100 },
                    plot_bgcolor: 'rgba(248,249,250,0.8)',
                    paper_bgcolor: 'white'
                  }}
                  style={{ width: '100%', height: '500px' }}
                  config={{
                    displayModeBar: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
                    responsive: true
                  }}
                />
              </div>
              
              {/* Chart Insights */}
              {response.analysis_insights && response.analysis_insights.data_interpretation && (
                <div style={{ 
                  marginTop: '1rem',
                  padding: '1rem',
                  backgroundColor: '#d1ecf1',
                  border: '1px solid #bee5eb',
                  borderRadius: '6px'
                }}>
                  <h6 style={{ color: '#0c5460', marginBottom: '0.5rem' }}>🔍 Data Insights</h6>
                  <ul style={{ margin: 0, paddingLeft: '1.5rem' }}>
                    {response.analysis_insights.data_interpretation.slice(0, 3).map((insight: string, idx: number) => (
                      <li key={idx} style={{ fontSize: '14px', marginBottom: '0.25rem', color: '#0c5460' }}>
                        {insight}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Insights Button */}
          <button 
            onClick={handleInsight}
            disabled={insightLoading}
            style={{
              padding: '8px 16px',
              backgroundColor: insightLoading ? '#6c757d' : '#17a2b8',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: insightLoading ? 'not-allowed' : 'pointer',
              marginTop: '1rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}
          >
            {insightLoading && (
              <div style={{
                width: '16px',
                height: '16px',
                border: '2px solid transparent',
                borderTop: '2px solid white',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite'
              }} />
            )}
            {insightLoading ? 'Generating...' : 'Generate Insights'}
          </button>

          {/* Insights Display */}
          {insight && (
            <div style={{ 
              marginTop: '1rem', 
              padding: '1rem', 
              background: '#e2f3ff', 
              border: '1px solid #b8daff',
              borderRadius: '4px' 
            }}>
              <h4 style={{ color: '#495057', marginBottom: '0.5rem' }}>💡 Insights</h4>
              <p>{insight}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default QueryPanel;
