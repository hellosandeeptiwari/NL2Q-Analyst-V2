import React, { useEffect, useState } from 'react';
import axios from 'axios';

function StatusPanel() {
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [manualConnect, setManualConnect] = useState(false);

  const fetchStatus = async () => {
    try {
      const response = await axios.get('http://localhost:8000/health');
      setStatus(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch status:', error);
      setStatus({
        connected: false,
        error: 'Connection failed',
        agents: { ready: false },
        tables: { count: 0 },
        database: { status: 'disconnected' }
      });
      setLoading(false);
    }
  };

  const connectToDatabase = async () => {
    setManualConnect(true);
    await fetchStatus();
    setManualConnect(false);
  };

  useEffect(() => {
    // Don't auto-fetch on first load, let user manually connect
    setLoading(false);
    setStatus({
      connected: false,
      database: { connected: false },
      agents: { ready: false },
      message: 'Click "Connect to Database" to check system status'
    });
  }, []);

  if (loading || manualConnect) return (
    <div style={{ background: '#fff', borderRadius: 8, boxShadow: '0 2px 8px #e2e8f0', padding: '1rem' }}>
      <h2 style={{ color: '#495057', marginBottom: '1rem' }}>System Status</h2>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <div style={{
          width: '16px',
          height: '16px',
          border: '2px solid transparent',
          borderTop: '2px solid #007bff',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite'
        }} />
        <span>{manualConnect ? 'Connecting...' : 'Loading status...'}</span>
      </div>
    </div>
  );

  const getStatusColor = (connected: boolean) => connected ? '#28a745' : '#dc3545';
  const getStatusText = (connected: boolean) => connected ? 'Connected' : 'Disconnected';

  return (
    <div style={{ background: '#fff', borderRadius: 8, boxShadow: '0 2px 8px #e2e8f0', padding: '1rem' }}>
      <h2 style={{ color: '#495057', marginBottom: '1rem' }}>System Status</h2>
      
      {!status?.database?.connected && (
        <div style={{ marginBottom: '1rem', textAlign: 'center' }}>
          <button
            onClick={connectToDatabase}
            disabled={manualConnect}
            style={{
              padding: '0.75rem 1.5rem',
              backgroundColor: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: manualConnect ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: '600'
            }}
          >
            {manualConnect ? 'Connecting...' : 'Connect to Database'}
          </button>
        </div>
      )}
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
        <div style={{ 
          padding: '0.75rem', 
          background: '#f8f9fa', 
          borderRadius: '6px',
          border: `2px solid ${getStatusColor(status?.connected === true)}`
        }}>
          <div style={{ fontWeight: '600', color: '#495057', marginBottom: '0.25rem' }}>Database</div>
          <div style={{ color: getStatusColor(status?.connected === true) }}>
            {getStatusText(status?.connected === true)}
          </div>
          {status?.latency_ms && (
            <div style={{ fontSize: '0.85rem', color: '#6c757d', marginTop: '0.25rem' }}>
              Latency: {status.latency_ms}ms
            </div>
          )}
        </div>

        <div style={{ 
          padding: '0.75rem', 
          background: '#f8f9fa', 
          borderRadius: '6px',
          border: `2px solid ${getStatusColor(status?.connected === true)}`
        }}>
          <div style={{ fontWeight: '600', color: '#495057', marginBottom: '0.25rem' }}>AI Agents</div>
          <div style={{ color: getStatusColor(status?.connected === true) }}>
            {status?.connected === true ? 'Ready' : 'Not Ready'}
          </div>
          {status?.connected && (
            <div style={{ fontSize: '0.85rem', color: '#6c757d', marginTop: '0.25rem' }}>
              System Ready
            </div>
          )}
        </div>
      </div>

      {status?.error && (
        <div style={{ 
          marginTop: '1rem', 
          padding: '0.75rem', 
          background: '#f8d7da', 
          color: '#721c24',
          borderRadius: '6px',
          fontSize: '0.9rem'
        }}>
          Error: {status.error}
        </div>
      )}

      {status?.message && !status?.error && (
        <div style={{ 
          marginTop: '1rem', 
          padding: '0.75rem', 
          background: '#d1ecf1', 
          color: '#0c5460',
          borderRadius: '6px',
          fontSize: '0.9rem',
          textAlign: 'center'
        }}>
          {status.message}
        </div>
      )}
      
      {status?.database?.connected && (
        <div style={{ 
          marginTop: '1rem', 
          fontSize: '0.8rem', 
          color: '#6c757d',
          textAlign: 'center'
        }}>
          Last updated: {new Date().toLocaleTimeString()}
        </div>
      )}
    </div>
  );
}

export default StatusPanel;
