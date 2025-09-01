import React, { useState } from 'react';
import StatusPanel from './components/StatusPanel';
import QueryPanel from './components/QueryPanelV2';
import ResultsGrid from './components/ResultsGrid';
import ChartPanel from './components/ChartPanel';
import DownloadButton from './components/DownloadButton';
import HelpPanel from './components/HelpPanel';
import QueryHistoryPanel from './components/QueryHistoryPanel';

// Configure API base URL
const API_BASE_URL = 'http://localhost:8003';

function App() {
  const [rows, setRows] = useState([]);
  const [jobId, setJobId] = useState('');
  const [plotlySpec, setPlotlySpec] = useState<any>(null);
  const [queryResponse, setQueryResponse] = useState<any>(null);
  const [insight, setInsight] = useState('');
  const [queryHistory, setQueryHistory] = useState<any[]>([]);

  // Function to handle query results from QueryPanel
  const handleQueryResults = (response: any) => {
    setQueryResponse(response);
    setRows(response?.rows || []);
    setPlotlySpec(response?.plotly_spec || null);
    setJobId(response?.job_id || '');
    
    // Add to query history
    if (response?.query) {
      setQueryHistory(prev => [...prev, {
        id: Date.now(),
        query: response.query,
        timestamp: new Date().toISOString(),
        table: response.table_used || 'Multiple',
        status: 'completed'
      }]);
    }
  };

  // Function to handle insights
  const handleInsightUpdate = (newInsight: string) => {
    setInsight(newInsight);
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <header style={{
        background: 'rgba(255, 255, 255, 0.1)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
        color: 'white',
        padding: '1rem 2rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        zIndex: 100
      }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{
            width: '48px',
            height: '48px',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginRight: '1rem',
            color: 'white',
            fontWeight: 'bold',
            fontSize: '20px',
            boxShadow: '0 8px 32px rgba(102, 126, 234, 0.4)'
          }}>
            NQ
          </div>
          <div>
            <h1 style={{ fontSize: '1.8rem', fontWeight: '700', marginBottom: '0.25rem' }}>
              NL2Q Agent
            </h1>
            <div style={{ fontSize: '0.9rem', opacity: 0.8, fontWeight: '400' }}>
              Enterprise Query Intelligence Platform
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{
            padding: '8px 16px',
            background: 'rgba(255, 255, 255, 0.1)',
            borderRadius: '20px',
            fontSize: '0.85rem',
            fontWeight: '500'
          }}>
            🚀 AI-Powered
          </div>
          <div style={{
            padding: '8px 16px',
            background: 'rgba(255, 255, 255, 0.1)',
            borderRadius: '20px',
            fontSize: '0.85rem',
            fontWeight: '500'
          }}>
            v2.0
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div style={{ display: 'flex', flex: 1 }}>
        {/* Sidebar */}
        <aside style={{
          width: '280px',
          background: 'rgba(255, 255, 255, 0.05)',
          backdropFilter: 'blur(10px)',
          borderRight: '1px solid rgba(255, 255, 255, 0.1)',
          padding: '2rem 1.5rem'
        }}>
          <div className="sidebar-nav">
            <nav>
              <div style={{ marginBottom: '1.5rem' }}>
                <h3 style={{ 
                  color: 'white', 
                  fontSize: '0.85rem', 
                  fontWeight: '600', 
                  textTransform: 'uppercase', 
                  letterSpacing: '0.05em',
                  marginBottom: '1rem',
                  opacity: 0.8
                }}>
                  Navigation
                </h3>
                <a href="#" className="nav-item active">
                  <span style={{ marginRight: '12px' }}>📊</span>
                  Dashboard
                </a>
                <a href="#" className="nav-item">
                  <span style={{ marginRight: '12px' }}>🔍</span>
                  Query Builder
                </a>
                <a href="#" className="nav-item">
                  <span style={{ marginRight: '12px' }}>📈</span>
                  Analytics
                </a>
                <a href="#" className="nav-item">
                  <span style={{ marginRight: '12px' }}>🕒</span>
                  Query History
                </a>
                <a href="#" className="nav-item">
                  <span style={{ marginRight: '12px' }}>⭐</span>
                  Favorites
                </a>
                <a href="#" className="nav-item">
                  <span style={{ marginRight: '12px' }}>⚙️</span>
                  Settings
                </a>
              </div>
            </nav>
          </div>
          
          <div style={{
            background: 'linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%)',
            borderRadius: '12px',
            padding: '1.5rem',
            border: '1px solid rgba(255,255,255,0.1)'
          }}>
            <h4 style={{ color: 'white', marginBottom: '0.75rem', fontSize: '0.9rem' }}>
              💡 Quick Tips
            </h4>
            <ul style={{ 
              listStyle: 'none', 
              padding: 0, 
              margin: 0,
              fontSize: '0.8rem',
              color: 'rgba(255,255,255,0.8)',
              lineHeight: 1.5
            }}>
              <li style={{ marginBottom: '0.5rem' }}>• Use natural language for queries</li>
              <li style={{ marginBottom: '0.5rem' }}>• AI suggests relevant tables</li>
              <li>• Get instant visualizations</li>
            </ul>
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="main-content" style={{ flex: 1 }}>
          <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
            <div className="card glass-card">
              <StatusPanel />
            </div>

            <div className="card glass-card">
              <QueryPanel 
                onQueryResults={handleQueryResults}
                onInsightUpdate={handleInsightUpdate}
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
              <div className="card glass-card">
                <ResultsGrid 
                  rows={rows}
                  columns={queryResponse?.columns || []}
                  jobId={jobId}
                />
              </div>
              <div className="card glass-card">
                <ChartPanel 
                  plotlySpec={plotlySpec}
                  response={queryResponse}
                />
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginTop: '2rem' }}>
              <div className="card glass-card">
                <QueryHistoryPanel 
                  queryHistory={queryHistory}
                />
              </div>
            </div>

            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', marginTop: '2rem' }}>
              <DownloadButton 
                jobId={jobId}
                rows={rows}
                disabled={!queryResponse}
              />
              <HelpPanel />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
