import React, { useState } from 'react';
import ConversationalQueryPanel from './components/ConversationalQueryPanel';

function App() {
  const [queryResponse, setQueryResponse] = useState<any>(null);
  const [insight, setInsight] = useState('');

  const handleQueryResults = (response: any) => {
    setQueryResponse(response);
  };

  const handleInsightUpdate = (newInsight: string) => {
    setInsight(newInsight);
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Header */}
      <header style={{
        background: 'linear-gradient(135deg, #0078d4 0%, #106ebe 100%)',
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
            ☁️
          </div>
          <div>
            <h1 style={{ 
              fontSize: '1.8rem', 
              fontWeight: '700', 
              marginBottom: '0.25rem',
              margin: 0
            }}>
              Azure Analytics Intelligence
            </h1>
            <div style={{ fontSize: '0.9rem', opacity: 0.8, fontWeight: '400' }}>
              Conversational Data Analysis Platform
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
            🤖 AI-Powered
          </div>
          <div style={{
            padding: '8px 16px',
            background: 'rgba(255, 255, 255, 0.1)',
            borderRadius: '20px',
            fontSize: '0.85rem',
            fontWeight: '500'
          }}>
            Enterprise v2.0
          </div>
        </div>
      </header>

      {/* Main Conversational Interface */}
      <main style={{ 
        flex: 1, 
        padding: '20px', 
        display: 'flex', 
        justifyContent: 'center',
        minHeight: 'calc(100vh - 100px)'
      }}>
        <div style={{ width: '100%', maxWidth: '1200px', height: '100%' }}>
          <ConversationalQueryPanel 
            onQueryResults={handleQueryResults}
            onInsightUpdate={handleInsightUpdate}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
