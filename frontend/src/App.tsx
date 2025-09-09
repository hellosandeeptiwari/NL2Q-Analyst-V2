import React, { useState } from 'react';
import './App.css';
import EnhancedPharmaChat from './components/EnhancedPharmaChat';
import DatabaseSettings from './components/DatabaseSettings';

function App() {
  const [currentPage, setCurrentPage] = useState('chat');

  return (
    <div className="App">
      {/* Keep chat component mounted to preserve state */}
      <div style={{ display: currentPage === 'chat' ? 'block' : 'none' }}>
        <EnhancedPharmaChat onNavigateToSettings={() => setCurrentPage('settings')} />
      </div>
      
      {/* Only mount settings when needed */}
      {currentPage === 'settings' && (
        <DatabaseSettings onNavigateBack={() => setCurrentPage('chat')} />
      )}
    </div>
  );
}

export default App;
