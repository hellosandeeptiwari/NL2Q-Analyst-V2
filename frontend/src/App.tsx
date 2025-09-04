import React, { useState } from 'react';
import './App.css';
import EnhancedPharmaChat from './components/EnhancedPharmaChat';
import DatabaseSettings from './components/DatabaseSettings';

function App() {
  const [currentPage, setCurrentPage] = useState('chat');

  const renderPage = () => {
    switch (currentPage) {
      case 'settings':
        return <DatabaseSettings onNavigateBack={() => setCurrentPage('chat')} />;
      case 'chat':
      default:
        return <EnhancedPharmaChat onNavigateToSettings={() => setCurrentPage('settings')} />;
    }
  };

  return (
    <div className="App">
      {renderPage()}
    </div>
  );
}

export default App;
