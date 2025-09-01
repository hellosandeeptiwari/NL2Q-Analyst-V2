import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface QueryHistoryPanelProps {
  queryHistory?: any[];
}

function QueryHistoryPanel({ queryHistory = [] }: QueryHistoryPanelProps) {
  const [loading, setLoading] = useState(false);

  // Use the passed queryHistory instead of fetching from API
  const history = queryHistory;

  const replayQuery = (query: string) => {
    // Emit event to QueryPanel to set the query
    window.dispatchEvent(new CustomEvent('replayQuery', { detail: query }));
  };

  if (loading) return <div style={{ textAlign: 'center', padding: '2rem' }}>Loading history...</div>;

  return (
    <div>
      <h2 style={{ color: '#495057', marginBottom: '1rem', fontSize: '1.25rem' }}>Query History</h2>
      {history.length === 0 ? (
        <div style={{ textAlign: 'center', color: '#6c757d', padding: '2rem' }}>No queries yet</div>
      ) : (
        <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
          {history.map((item, index) => (
            <div key={index} style={{
              border: '1px solid #e9ecef',
              borderRadius: '4px',
              padding: '1rem',
              marginBottom: '0.5rem',
              background: '#fff'
            }}>
              <div style={{ fontSize: '0.9rem', color: '#6c757d', marginBottom: '0.5rem' }}>
                {new Date(item.timestamp).toLocaleString()}
              </div>
              <div style={{ marginBottom: '0.5rem', fontFamily: 'monospace', fontSize: '0.9rem' }}>
                {item.nl_query}
              </div>
              <button
                onClick={() => replayQuery(item.nl_query)}
                className="btn-primary"
                style={{ fontSize: '0.8rem', padding: '0.25rem 0.75rem' }}
              >
                Replay
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default QueryHistoryPanel;
