import React, { useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';

function QueryPanel() {
  const [nl, setNl] = useState('');
  const [jobId, setJobId] = useState('');
  const [dbType, setDbType] = useState('sqlite');
  const [response, setResponse] = useState<any>(null);
  const [error, setError] = useState('');
  const [insight, setInsight] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    try {
      const res = await axios.post('http://localhost:8003/query', {
        natural_language: nl,
        job_id: jobId || `job_${Date.now()}`,
        db_type: dbType
      });
      setResponse(res.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Error running query');
    }
  };

  const handleInsight = async () => {
    if (!response) return;
    try {
      const res = await axios.post('http://localhost:8003/insights', {
        location: response.location,
        query: nl
      });
      setInsight(res.data.insight);
    } catch (err: any) {
      setError('Error generating insights');
    }
  };

  return (
    <div>
      <h2 style={{ color: '#495057', marginBottom: '1.5rem', fontSize: '1.5rem' }}>Natural Language Query</h2>
      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: '300px' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#495057' }}>
            Query
          </label>
          <input
            type="text"
            value={nl}
            onChange={e => setNl(e.target.value)}
            placeholder="Enter your question..."
            style={{ width: '100%' }}
            required
          />
        </div>
        <div style={{ minWidth: '150px' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#495057' }}>
            Database
          </label>
          <select
            value={dbType}
            onChange={e => setDbType(e.target.value)}
            style={{ width: '100%' }}
          >
            <option value="sqlite">SQLite</option>
            <option value="postgres">PostgreSQL</option>
            <option value="snowflake">Snowflake</option>
            <option value="azure_sql">Azure SQL</option>
          </select>
        </div>
        <div style={{ minWidth: '150px' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#495057' }}>
            Job ID
          </label>
          <input
            type="text"
            value={jobId}
            onChange={e => setJobId(e.target.value)}
            placeholder="Optional"
            style={{ width: '100%' }}
          />
        </div>
        <button type="submit" className="btn-primary" style={{ height: '40px', padding: '0 2rem' }}>
          Execute Query
        </button>
      </form>
      {error && <div style={{ color: '#dc3545', marginTop: '1rem', padding: '0.75rem', background: '#f8d7da', borderRadius: '4px', border: '1px solid #f5c6cb' }}>{error}</div>}
      {response && (
        <div style={{ marginTop: '2rem' }}>
          <h3 style={{ color: '#495057', marginBottom: '1rem' }}>Generated SQL</h3>
          <pre style={{ background: '#f8f9fa', padding: '1rem', borderRadius: '4px', border: '1px solid #e9ecef', overflow: 'auto', fontSize: '0.9rem' }}>{response.sql}</pre>
          {response.suggestions && response.suggestions.length > 0 && (
            <div style={{ marginTop: '1.5rem' }}>
              <h4 style={{ color: '#495057', marginBottom: '0.5rem' }}>Query Suggestions</h4>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                {response.suggestions.map((suggestion: string, index: number) => (
                  <button
                    key={index}
                    onClick={() => setNl(suggestion)}
                    style={{
                      padding: '0.5rem 1rem',
                      background: '#e9ecef',
                      border: '1px solid #dee2e6',
                      borderRadius: '20px',
                      cursor: 'pointer',
                      fontSize: '0.9rem',
                      color: '#495057'
                    }}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}
          {response.plotly_spec && response.plotly_spec.data && (
            <div style={{ marginTop: '1.5rem' }}>
              <h4 style={{ color: '#495057', marginBottom: '0.5rem' }}>Visualization</h4>
              <Plot data={response.plotly_spec.data} layout={response.plotly_spec.layout} />
            </div>
          )}
          <div style={{ marginTop: '1.5rem' }}>
            <button onClick={handleInsight} className="btn-primary">
              Generate Insights
            </button>
          </div>
          {insight && (
            <div style={{ marginTop: '1.5rem' }}>
              <h4 style={{ color: '#495057', marginBottom: '0.5rem' }}>AI Insights</h4>
              <div style={{ background: '#e7f3ff', padding: '1rem', borderRadius: '4px', border: '1px solid #b8daff' }}>
                {insight}
              </div>
            </div>
          )}
          {response.bias_report && (
            <div style={{ marginTop: '1.5rem' }}>
              <h4 style={{ color: '#495057', marginBottom: '0.5rem' }}>Bias Analysis</h4>
              <div style={{ background: '#fff3cd', padding: '1rem', borderRadius: '4px', border: '1px solid #ffeaa7' }}>
                {response.bias_report}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default QueryPanel;
